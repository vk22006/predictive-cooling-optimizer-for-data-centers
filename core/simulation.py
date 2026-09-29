# SPDX-License-Identifier: MIT
"""
Simulation — deterministic one-step state transition for the live
dashboard.

Separates:
- simulation state  (``SimulationState``)
- simulation clock  (caller-controlled — no ``time.sleep``)
- input data        (pre-engineered CSV rows)
- prediction        (via ``core.prediction``)
- optimization      (via ``core.optimization``, optional per-step)
- visualisation     (not here — handled by the UI layer)

The caller drives the loop::

    state = create_initial_state(df)
    while state.is_running:
        state, frame = simulation_step(state, df, store)
        # send ``frame`` to the UI / SSE stream / websocket
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import pandas as pd

from core.model_loader import ModelStore
from core.prediction import predict_energy_from_row, predict_temp_from_row
from core.schemas import SimulationFrame, SimulationState


def create_initial_state(df: pd.DataFrame) -> SimulationState:
    """Create a fresh simulation state ready for stepping.

    Parameters:
        df: the pre-engineered DataFrame (e.g. from
            ``sample_test_data.csv``).  Its length determines the total
            number of simulation steps.

    Returns:
        A ``SimulationState`` with ``is_running=True`` and
        ``current_index=0``.
    """
    if df.empty:
        raise ValueError("Cannot create simulation from an empty DataFrame.")
    return SimulationState(current_index=0, is_running=True, history=[])


def simulation_step(
    state: SimulationState,
    df: pd.DataFrame,
    store: Optional[ModelStore] = None,
) -> Tuple[SimulationState, Optional[SimulationFrame]]:
    """Advance the simulation by one deterministic step.

    1. Reads row ``state.current_index`` from ``df``.
    2. Predicts energy (using the energy model's feature set from the
       CSV row).
    3. Predicts temperature (injecting predicted energy into the temp
       model's feature set).
    4. Computes proxy metrics (lagged energy/temp from the CSV).
    5. Returns the updated state and the frame.

    If the simulation has already consumed all rows, sets
    ``state.is_running = False`` and returns ``(state, None)``.

    The caller controls timing.  No ``time.sleep`` or blocking occurs.

    Parameters:
        state: current simulation state.
        df: the full pre-engineered DataFrame.
        store: ModelStore; defaults to singleton.

    Returns:
        (next_state, frame) — frame is ``None`` when the simulation ends.
    """
    if not state.is_running or state.current_index >= len(df):
        state = copy.copy(state)
        state.is_running = False
        return state, None

    store = store or ModelStore.get()
    idx = state.current_index
    row = df.iloc[idx]

    # --- Energy prediction ---
    pred_energy = predict_energy_from_row(row, store)

    # --- Temperature prediction ---
    pred_temp = predict_temp_from_row(row, pred_energy, store)

    # --- Proxy / lagged values from the CSV ---
    # These are NOT ground-truth actuals.  They are the best
    # approximation available in the dataset.
    lagged_energy = _safe_float(row, "Energy_Lag_1", fallback=0.0)
    lagged_outside_temp = _safe_float(row, "OutsideTemp_Lag_1", fallback=0.0)

    # --- Potential savings (relative to lagged proxy) ---
    if lagged_energy > 0:
        potential_savings_pct = (lagged_energy - pred_energy) / lagged_energy
    else:
        potential_savings_pct = 0.0

    frame = SimulationFrame(
        step_index=idx,
        predicted_energy_kwh=pred_energy,
        lagged_energy_kwh=lagged_energy,
        predicted_temp_c=pred_temp,
        lagged_outside_temp_f=lagged_outside_temp,
        potential_savings_pct=potential_savings_pct,
    )

    # --- Advance state ---
    next_state = copy.copy(state)
    next_state.current_index = idx + 1
    next_state.history = state.history + [frame]
    if next_state.current_index >= len(df):
        next_state.is_running = False

    return next_state, frame


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _safe_float(row: pd.Series, col: str, fallback: float) -> float:
    """Read a float from a Series, returning *fallback* only if the
    column does not exist.  Does not silently mask NaN — NaN is
    propagated if the column exists but contains NaN."""
    if col in row.index:
        val = row[col]
        return float(val)
    return fallback
