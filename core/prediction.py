# SPDX-License-Identifier: MIT
"""
Prediction service — clean Python functions for energy, temperature,
and combined prediction.

No Streamlit or FastAPI dependencies.  Consumes the ModelStore singleton
and the canonical feature-engineering pipeline.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.feature_engineering import (
    FeatureEngineeringError,
    build_energy_features,
    build_energy_features_from_row,
    build_temp_features,
    build_temp_features_from_row,
    validate_feature_vector,
)
from core.model_loader import ModelStore
from core.schemas import (
    CombinedPrediction,
    EnergyPrediction,
    SensorReading,
    TemperaturePrediction,
)


class PredictionError(Exception):
    """Raised when a model prediction fails."""


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def predict_energy(
    reading: SensorReading,
    history_df: pd.DataFrame,
    store: Optional[ModelStore] = None,
) -> EnergyPrediction:
    """Predict energy consumption (kWh) from a sensor reading.

    Parameters:
        reading: current sensor measurements.
        history_df: DataFrame of preceding rows (≥12 rows) with columns
            ``Energy``, ``Building Load (RT)``,
            ``Outside Temperature (F)``, ``Cooling Water Temperature (C)``.
        store: optional ModelStore; defaults to singleton.

    Returns:
        EnergyPrediction with the predicted value.

    Raises:
        FeatureEngineeringError: if the feature vector cannot be built.
        PredictionError: if the model inference itself fails.
    """
    store = store or ModelStore.get()
    X = build_energy_features(reading, history_df)
    validate_feature_vector(X, store.energy_feature_names, "energy")
    return EnergyPrediction(predicted_energy_kwh=_infer(store.energy_model, X))


def predict_temperature(
    reading: SensorReading,
    history_df: pd.DataFrame,
    predicted_energy_kwh: float,
    store: Optional[ModelStore] = None,
) -> TemperaturePrediction:
    """Predict temperature (°C) one hour ahead.

    The temperature model requires the *predicted* energy as an input
    feature (``Chiller Energy Consumption (kWh)``), reflecting the
    chained-prediction design of the original system.

    Parameters:
        reading: current sensor measurements.
        history_df: same as for ``predict_energy``.
        predicted_energy_kwh: output of the energy model for this reading.
        store: optional ModelStore; defaults to singleton.

    Returns:
        TemperaturePrediction with the predicted value.
    """
    store = store or ModelStore.get()
    X = build_temp_features(reading, history_df, predicted_energy_kwh)
    validate_feature_vector(X, store.temp_feature_names, "temperature")
    return TemperaturePrediction(
        predicted_temp_c=_infer(store.temp_model, X)
    )


def predict_combined(
    reading: SensorReading,
    history_df: pd.DataFrame,
    store: Optional[ModelStore] = None,
) -> CombinedPrediction:
    """Run the full chained prediction: energy → temperature.

    Returns both predictions together.
    """
    energy_result = predict_energy(reading, history_df, store)
    temp_result = predict_temperature(
        reading, history_df, energy_result.predicted_energy_kwh, store
    )
    return CombinedPrediction(
        predicted_energy_kwh=energy_result.predicted_energy_kwh,
        predicted_temp_c=temp_result.predicted_temp_c,
    )


# -----------------------------------------------------------------------
# Pre-engineered row helpers  (used by simulation / dashboard)
# -----------------------------------------------------------------------

def predict_energy_from_row(
    row: pd.Series,
    store: Optional[ModelStore] = None,
) -> float:
    """Predict energy from a pre-engineered CSV row (e.g.
    ``sample_test_data.csv``).

    The row must contain all columns in ``store.energy_feature_names``.
    """
    store = store or ModelStore.get()
    X = build_energy_features_from_row(row, store.energy_feature_names)
    return _infer(store.energy_model, X)


def predict_temp_from_row(
    row: pd.Series,
    predicted_energy_kwh: float,
    store: Optional[ModelStore] = None,
) -> float:
    """Predict temperature from a pre-engineered CSV row, injecting the
    predicted energy value for the ``Chiller Energy Consumption (kWh)``
    column that is not present in the CSV.
    """
    store = store or ModelStore.get()
    X = build_temp_features_from_row(
        row, store.temp_feature_names, predicted_energy_kwh
    )
    return _infer(store.temp_model, X)


# -----------------------------------------------------------------------
# Internal
# -----------------------------------------------------------------------

def _infer(model: object, X: pd.DataFrame) -> float:
    """Run model.predict and return a scalar float."""
    try:
        return float(model.predict(X)[0])  # type: ignore[union-attr]
    except Exception as exc:
        raise PredictionError(
            f"Model inference failed: {exc}"
        ) from exc
