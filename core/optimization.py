# SPDX-License-Identifier: MIT
"""
Cooling optimization service — grid-search over chilled-water-rate
adjustments.

Extracted from ``notebook/optimization_engine.ipynb`` and
``notebook/system_integration_and_testing.ipynb``.

Algorithm documentation
-----------------------
**Objective**: minimise predicted chiller energy consumption (kWh) while
keeping predicted temperature within acceptable bounds.

**Control variable**: Chilled Water Rate (L/sec) — adjusted by an additive
delta applied to the current operating value.

**Search space**: a discrete grid of adjustments to the *raw* feature
value.  The notebook used adjustments of ±{0.02, 0.05, 0.10} on
*normalised* data (values in [0, 1]).  However, the deployed models
were trained on *raw-scale* data (chilled water rate in
~[72, 142] L/sec), meaning those small adjustments on normalised data
correspond to ~±1–7 L/sec on raw data.

To match the notebook's intent while operating on raw-scale features,
this module uses a default search grid of:

    [-5.0, -2.0, -1.0, 0.0, +1.0, +2.0, +5.0]  L/sec

These are comparable to the normalised-space steps the notebook used.
The grid is configurable.

**Constraints**:
- Chilled-water rate must stay within the training-data range
  [72.40, 141.50] L/sec.
- Temperature prediction is computed at each candidate point but is
  currently *not* used as a hard constraint (consistent with the
  notebook, which checked ``if test_energy < best_energy`` without a
  temperature gate).

**Selection criterion**: the candidate with the lowest predicted energy
that keeps the chilled-water rate within its valid range.

**Known limitations / assumptions documented from the notebook**:
1. Only one control variable is adjusted (chilled-water rate).  Other
   controllable inputs (cooling-water temperature, building load) are
   held fixed.
2. The notebook's grid-search operates on a single input row, not over a
   time horizon.  No dynamic / multi-step optimisation is performed.
3. The notebook clips to [0, 1] (normalised), but the models here
   operate on raw data.  We clip to the observed training-data range
   instead.
4. Temperature constraints are *not* enforced as hard constraints in the
   notebook; we document the predicted temperature but do not reject
   candidates solely on temperature.
5. The cost-savings calculation is purely extrapolative and assumes a
   constant cost-per-kWh.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from core.config import (
    CHILLED_WATER_MAX,
    CHILLED_WATER_MIN,
    ENERGY_MODEL_FEATURES,
    TEMP_MODEL_FEATURES,
)
from core.model_loader import ModelStore
from core.schemas import CandidatePoint, OptimizationResult


# Default search grid: additive adjustments in L/sec.
DEFAULT_ADJUSTMENTS: List[float] = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]


def optimize_cooling(
    energy_feature_row: pd.DataFrame,
    store: Optional[ModelStore] = None,
    adjustments: Optional[List[float]] = None,
    include_temp: bool = True,
) -> OptimizationResult:
    """Find the chilled-water-rate adjustment that minimises predicted
    energy consumption.

    Parameters:
        energy_feature_row: a 1-row DataFrame matching the energy model's
            feature ordering (46 columns, same order as
            ``ENERGY_MODEL_FEATURES``).
        store: ModelStore; defaults to singleton.
        adjustments: list of additive deltas (L/sec) to try.
            Defaults to ``DEFAULT_ADJUSTMENTS``.
        include_temp: if True, also predict temperature at each candidate
            point (requires building a temp-model input).

    Returns:
        OptimizationResult with baseline, optimised, savings, and the
        full list of candidate points evaluated.
    """
    store = store or ModelStore.get()
    if adjustments is None:
        adjustments = DEFAULT_ADJUSTMENTS

    if energy_feature_row.shape[0] != 1:
        raise ValueError(
            f"Expected a 1-row DataFrame, got {energy_feature_row.shape[0]} "
            f"rows."
        )

    base_row = energy_feature_row.copy()
    cwr_col = "Chilled Water Rate (L/sec)"
    cwr_idx = ENERGY_MODEL_FEATURES.index(cwr_col)

    current_cwr = float(base_row.iloc[0, cwr_idx])
    baseline_energy = float(store.energy_model.predict(base_row)[0])

    # Optionally compute baseline temperature
    baseline_temp: Optional[float] = None
    if include_temp:
        baseline_temp = _predict_temp_from_energy_row(
            base_row, baseline_energy, store
        )

    candidates: List[CandidatePoint] = []
    best_energy = baseline_energy
    best_idx = -1

    for i, adj in enumerate(adjustments):
        test_cwr = current_cwr + adj

        # Enforce training-data range
        test_cwr = float(np.clip(test_cwr, CHILLED_WATER_MIN, CHILLED_WATER_MAX))

        # Build test row
        test_row = base_row.copy()
        test_row.iloc[0, cwr_idx] = test_cwr

        pred_energy = float(store.energy_model.predict(test_row)[0])

        # Optionally predict temp at this operating point
        pred_temp: Optional[float] = None
        if include_temp:
            pred_temp = _predict_temp_from_energy_row(
                test_row, pred_energy, store
            )

        candidate = CandidatePoint(
            chilled_water_rate_adjustment=adj,
            predicted_energy_kwh=pred_energy,
            predicted_temp_c=pred_temp,
            is_selected=False,
        )
        candidates.append(candidate)

        if pred_energy < best_energy:
            best_energy = pred_energy
            best_idx = i

    # Mark the selected candidate
    if best_idx >= 0:
        # Replace the candidate with is_selected=True (frozen dataclass)
        c = candidates[best_idx]
        candidates[best_idx] = CandidatePoint(
            chilled_water_rate_adjustment=c.chilled_water_rate_adjustment,
            predicted_energy_kwh=c.predicted_energy_kwh,
            predicted_temp_c=c.predicted_temp_c,
            is_selected=True,
        )

    savings = baseline_energy - best_energy
    savings_pct = (savings / baseline_energy * 100) if baseline_energy > 0 else 0.0

    # Generate suggestion
    if best_idx >= 0 and savings > 0.01:
        best_adj = candidates[best_idx].chilled_water_rate_adjustment
        suggestion = (
            f"Adjust chilled-water rate by {best_adj:+.1f} L/sec "
            f"to save {savings:.2f} kWh ({savings_pct:.1f}%)."
        )
    else:
        suggestion = "Current settings are already near-optimal."

    return OptimizationResult(
        baseline_energy_kwh=baseline_energy,
        optimized_energy_kwh=best_energy,
        energy_savings_kwh=savings,
        savings_percent=savings_pct,
        baseline_temp_c=baseline_temp,
        suggestion=suggestion,
        candidates=candidates,
    )


# -----------------------------------------------------------------------
# Internal helper
# -----------------------------------------------------------------------

def _predict_temp_from_energy_row(
    energy_row: pd.DataFrame,
    predicted_energy: float,
    store: ModelStore,
) -> float:
    """Build a temperature-model input from an energy-model feature row.

    Maps the shared engineered features and injects the predicted energy
    into the ``Chiller Energy Consumption (kWh)`` slot.
    """
    energy_cols = list(energy_row.columns)
    values = energy_row.iloc[0]

    temp_row_dict = {}
    for col in TEMP_MODEL_FEATURES:
        if col == "Chiller Energy Consumption (kWh)":
            temp_row_dict[col] = predicted_energy
        elif col in energy_cols:
            temp_row_dict[col] = float(values[col])
        elif col == "Building Load (RT)" and "Building Load (RT)" in energy_cols:
            temp_row_dict[col] = float(values["Building Load (RT)"])
        else:
            # This should not happen if both feature sets share
            # 45 out of 46 columns.  Raise instead of zero-filling.
            raise ValueError(
                f"Cannot map energy feature row to temp feature row: "
                f"column '{col}' not found and no mapping rule exists."
            )

    temp_df = pd.DataFrame([temp_row_dict], columns=TEMP_MODEL_FEATURES)
    return float(store.temp_model.predict(temp_df)[0])
