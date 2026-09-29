# SPDX-License-Identifier: MIT
"""
Canonical feature engineering pipeline.

Produces the 46-feature vectors expected by the energy and temperature
XGBoost models.  All logic is extracted from the original Streamlit
``create_feature_vector()`` in ``pages/3_Manual_Prediction.py`` and
validated against the trained model artefacts.

Key design decisions
--------------------
* **No silent zero-fill.**  If a required feature cannot be calculated
  (e.g. not enough history rows), a ``FeatureEngineeringError`` is raised
  instead of silently inserting 0.0.
* **Deterministic ordering.**  The output DataFrame column order always
  matches the model's ``feature_names_in_``.
* **Two separate builders** for energy vs temperature, because the two
  models differ in their raw-input slot (energy expects
  ``Cooling Water Temperature (C)``; temp expects
  ``Chiller Energy Consumption (kWh)``).
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import (
    ENERGY_MODEL_FEATURES,
    LAG_DEPTHS,
    MIN_HISTORY_ROWS,
    TEMP_MODEL_FEATURES,
)
from core.schemas import SensorReading


class FeatureEngineeringError(Exception):
    """Raised when the feature-engineering pipeline cannot produce a valid
    feature vector."""


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _ensure_history_columns(hist: pd.DataFrame) -> pd.DataFrame:
    """Ensure the history DataFrame has the columns required for lag and
    rolling calculations.  Raises rather than filling with zeros."""
    required = [
        "Energy",
        "Building Load (RT)",
        "Outside Temperature (F)",
        "Cooling Water Temperature (C)",
    ]
    missing = [c for c in required if c not in hist.columns]
    if missing:
        raise FeatureEngineeringError(
            f"History DataFrame is missing columns required for "
            f"lag / rolling calculations: {missing}"
        )
    return hist


def _compute_lags(
    hist_with_current: pd.DataFrame,
    last_idx: int,
) -> Dict[str, float]:
    """Compute the 16 lag features (4 sensors × 4 depths)."""
    features: Dict[str, float] = {}

    sensor_map = {
        "Energy": "Energy",
        "BuildingLoad": "Building Load (RT)",
        "OutsideTemp": "Outside Temperature (F)",
        "CoolingWaterTemp": "Cooling Water Temperature (C)",
    }

    for depth in LAG_DEPTHS:
        row_idx = last_idx - depth
        if row_idx < 0:
            raise FeatureEngineeringError(
                f"Not enough history to compute lag-{depth} "
                f"(need at least {depth + 1} rows, have {last_idx + 1})."
            )
        for short_name, col_name in sensor_map.items():
            key = f"{short_name}_Lag_{depth}"
            features[key] = float(hist_with_current.iloc[row_idx][col_name])

    return features


def _compute_rolling(
    hist_with_current: pd.DataFrame,
    last_idx: int,
) -> Dict[str, float]:
    """Compute the 12 rolling-window features.

    Windows look backward from (but excluding) the current row, matching
    the original pipeline: ``hist_with_current.iloc[-window-1:-1]``.
    """
    features: Dict[str, float] = {}

    windows = [3, 6, 12]
    for w in windows:
        start = last_idx - w
        end = last_idx  # exclusive of current row
        if start < 0:
            raise FeatureEngineeringError(
                f"Not enough history for rolling-{w} "
                f"(need {w} preceding rows)."
            )
        window_slice = hist_with_current.iloc[start:end]

        # Energy rolling avg + std
        e_vals = window_slice["Energy"]
        features[f"Energy_RollingAvg_{w}"] = float(e_vals.mean())
        features[f"Energy_RollingStd_{w}"] = (
            float(e_vals.std()) if len(e_vals) > 1 else 0.0
        )

        # Building Load rolling avg
        features[f"BuildingLoad_RollingAvg_{w}"] = float(
            window_slice["Building Load (RT)"].mean()
        )

        # Outside Temp rolling avg
        features[f"OutsideTemp_RollingAvg_{w}"] = float(
            window_slice["Outside Temperature (F)"].mean()
        )

    return features


def _compute_cyclical(ts: datetime.datetime) -> Dict[str, float]:
    """Compute the 6 cyclical temporal encoding features."""
    hour = ts.hour
    dow = ts.weekday()
    month = ts.month
    return {
        "Hour_Sin": np.sin(2 * np.pi * hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24),
        "DayOfWeek_Sin": np.sin(2 * np.pi * dow / 7),
        "DayOfWeek_Cos": np.cos(2 * np.pi * dow / 7),
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
    }


def _compute_interactions(
    building_load: float,
    outside_temp: float,
    chilled_water_rate: float,
    humidity: float,
    hour: int,
) -> Dict[str, float]:
    """Compute the 4 interaction features."""
    return {
        "Load_Temp_Interaction": building_load * outside_temp,
        "ChilledWater_Load_Interaction": chilled_water_rate * building_load,
        "Temp_Humidity_Interaction": outside_temp * humidity,
        "Hour_Load_Interaction": hour * building_load,
    }


# -----------------------------------------------------------------------
# Core feature-vector builders
# -----------------------------------------------------------------------

def _build_engineered_features(
    reading: SensorReading,
    history_df: pd.DataFrame,
) -> Dict[str, float]:
    """Build the full set of engineered features (lags, rolling, cyclical,
    interactions) from a sensor reading and its preceding history.

    Returns a dict keyed by feature name.  Does **not** include the
    model-specific raw slot (``Cooling Water Temperature (C)`` vs
    ``Chiller Energy Consumption (kWh)``); callers add those.
    """
    hist = _ensure_history_columns(history_df.copy())

    # Build a raw row to append to history (for lag/rolling indexing).
    raw_row = {
        "Chilled Water Rate (L/sec)": reading.chilled_water_rate,
        "Cooling Water Temperature (C)": reading.cooling_water_temp,
        "Building Load (RT)": reading.building_load,
        "Outside Temperature (F)": reading.outside_temp,
        "Dew Point (F)": reading.dew_point,
        "Humidity (%)": reading.humidity,
        "Wind Speed (mph)": reading.wind_speed,
        "Pressure (in)": reading.pressure,
        "Energy": reading.chiller_energy,
    }
    hist_with_current = pd.concat(
        [hist, pd.DataFrame([raw_row])], ignore_index=True
    )
    last_idx = len(hist_with_current) - 1

    if last_idx < MIN_HISTORY_ROWS - 1:
        raise FeatureEngineeringError(
            f"Need at least {MIN_HISTORY_ROWS} total rows "
            f"(history + current) for rolling-12. Have {last_idx + 1}."
        )

    features: Dict[str, float] = {}

    # Raw inputs common to both models
    features["Chilled Water Rate (L/sec)"] = reading.chilled_water_rate
    features["Building Load (RT)"] = reading.building_load
    features["Outside Temperature (F)"] = reading.outside_temp
    features["Dew Point (F)"] = reading.dew_point
    features["Humidity (%)"] = reading.humidity
    features["Wind Speed (mph)"] = reading.wind_speed
    features["Pressure (in)"] = reading.pressure

    # Lag features (16)
    features.update(_compute_lags(hist_with_current, last_idx))

    # Rolling features (12)
    features.update(_compute_rolling(hist_with_current, last_idx))

    # Cyclical features (6)
    features.update(_compute_cyclical(reading.timestamp))

    # Interaction features (4)
    features.update(
        _compute_interactions(
            building_load=reading.building_load,
            outside_temp=reading.outside_temp,
            chilled_water_rate=reading.chilled_water_rate,
            humidity=reading.humidity,
            hour=reading.timestamp.hour,
        )
    )

    return features


def build_energy_features(
    reading: SensorReading,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a 1-row DataFrame with the 46 features the *energy* model
    expects, in the exact column order from
    ``config.ENERGY_MODEL_FEATURES``.

    The energy model's differentiating raw column is
    ``Cooling Water Temperature (C)``.

    Raises:
        FeatureEngineeringError: if any required feature cannot be
            computed from the provided data.
    """
    features = _build_engineered_features(reading, history_df)

    # Energy-model specific raw slot
    features["Cooling Water Temperature (C)"] = reading.cooling_water_temp

    return _assemble(features, ENERGY_MODEL_FEATURES, "energy")


def build_temp_features(
    reading: SensorReading,
    history_df: pd.DataFrame,
    predicted_energy_kwh: float,
) -> pd.DataFrame:
    """Build a 1-row DataFrame with the 46 features the *temperature*
    model expects, in the exact column order from
    ``config.TEMP_MODEL_FEATURES``.

    The temperature model's differentiating raw column is
    ``Chiller Energy Consumption (kWh)`` — this must be the *predicted*
    energy from the energy model (chained prediction pattern).

    Raises:
        FeatureEngineeringError: if any required feature cannot be
            computed from the provided data.
    """
    features = _build_engineered_features(reading, history_df)

    # Temp-model specific raw slot
    features["Chiller Energy Consumption (kWh)"] = predicted_energy_kwh

    return _assemble(features, TEMP_MODEL_FEATURES, "temperature")


def build_energy_features_from_row(
    row: pd.Series,
    feature_names: List[str],
) -> pd.DataFrame:
    """Build an energy-model input from a pre-engineered CSV row.

    Used by the simulation path where ``sample_test_data.csv`` already
    contains all 46 engineered columns.  Validates that every required
    feature exists in the row.

    Raises:
        FeatureEngineeringError: if any required feature is absent.
    """
    missing = [c for c in feature_names if c not in row.index]
    if missing:
        raise FeatureEngineeringError(
            f"Pre-engineered row is missing features required by the "
            f"energy model: {missing}"
        )
    return pd.DataFrame([row[feature_names].values], columns=feature_names)


def build_temp_features_from_row(
    row: pd.Series,
    feature_names: List[str],
    predicted_energy_kwh: float,
) -> pd.DataFrame:
    """Build a temperature-model input from a pre-engineered CSV row.

    The CSV does not contain ``Chiller Energy Consumption (kWh)``; this
    function inserts the predicted energy in its place.  All other
    features must already exist in the row.

    Raises:
        FeatureEngineeringError: if any required feature (other than the
            injected energy column) is absent.
    """
    injected_col = "Chiller Energy Consumption (kWh)"
    required_from_row = [c for c in feature_names if c != injected_col]
    missing = [c for c in required_from_row if c not in row.index]
    if missing:
        raise FeatureEngineeringError(
            f"Pre-engineered row is missing features required by the "
            f"temperature model: {missing}"
        )
    values = {}
    for col in feature_names:
        if col == injected_col:
            values[col] = predicted_energy_kwh
        else:
            values[col] = row[col]
    return pd.DataFrame([values], columns=feature_names)


# -----------------------------------------------------------------------
# Assembly & validation
# -----------------------------------------------------------------------

def _assemble(
    features: Dict[str, float],
    expected_order: List[str],
    model_label: str,
) -> pd.DataFrame:
    """Assemble a features dict into a 1-row DataFrame in the exact column
    order the model expects.

    Raises ``FeatureEngineeringError`` if any expected feature is missing
    (no silent zero-fill).
    """
    missing = [col for col in expected_order if col not in features]
    if missing:
        raise FeatureEngineeringError(
            f"Cannot build {model_label}-model feature vector. "
            f"Missing features: {missing}"
        )
    row = {col: features[col] for col in expected_order}
    return pd.DataFrame([row], columns=expected_order)


def validate_feature_vector(
    df: pd.DataFrame,
    expected_features: List[str],
    model_label: str = "model",
) -> None:
    """Validate that a feature-vector DataFrame matches expectations.

    Checks:
    1. Correct number of features.
    2. No NaN values.
    3. Column names match in order.

    Raises ``FeatureEngineeringError`` on any violation.
    """
    actual_cols = list(df.columns)
    if len(actual_cols) != len(expected_features):
        raise FeatureEngineeringError(
            f"{model_label}: expected {len(expected_features)} features, "
            f"got {len(actual_cols)}."
        )
    mismatches = [
        (i, a, e)
        for i, (a, e) in enumerate(zip(actual_cols, expected_features))
        if a != e
    ]
    if mismatches:
        detail = "; ".join(
            f"[{i}] got={a!r} expected={e!r}" for i, a, e in mismatches
        )
        raise FeatureEngineeringError(
            f"{model_label}: column ordering mismatch: {detail}"
        )
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        raise FeatureEngineeringError(
            f"{model_label}: NaN values in features: {nan_cols}"
        )
