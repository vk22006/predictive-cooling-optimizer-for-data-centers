# SPDX-License-Identifier: MIT
"""Tests for core.prediction — energy, temperature, and combined."""

import pandas as pd
import pytest

from core.model_loader import ModelStore
from core.prediction import (
    PredictionError,
    predict_combined,
    predict_energy,
    predict_energy_from_row,
    predict_temp_from_row,
    predict_temperature,
)
from core.schemas import (
    CombinedPrediction,
    EnergyPrediction,
    SensorReading,
    TemperaturePrediction,
)


class TestEnergyPrediction:
    """Verify energy prediction interface and outputs."""

    def test_returns_energy_prediction(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        result = predict_energy(sample_reading, history_df, store)
        assert isinstance(result, EnergyPrediction)

    def test_energy_is_positive(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        result = predict_energy(sample_reading, history_df, store)
        assert result.predicted_energy_kwh > 0

    def test_energy_in_reasonable_range(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        """Energy should be within the observed training range
        (~50–250 kWh)."""
        result = predict_energy(sample_reading, history_df, store)
        assert 30 < result.predicted_energy_kwh < 300

    def test_energy_from_row_matches_direct(
        self,
        sample_df: pd.DataFrame,
        store: ModelStore,
    ):
        """The pre-engineered-row path should produce the same energy
        prediction as calling the model directly."""
        row = sample_df.iloc[0]
        pred = predict_energy_from_row(row, store)
        # Compare against direct model call
        X = sample_df[store.energy_feature_names].iloc[0:1]
        direct = float(store.energy_model.predict(X)[0])
        assert abs(pred - direct) < 1e-6


class TestTemperaturePrediction:
    """Verify temperature prediction interface and outputs."""

    def test_returns_temp_prediction(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        energy = predict_energy(sample_reading, history_df, store)
        result = predict_temperature(
            sample_reading,
            history_df,
            energy.predicted_energy_kwh,
            store,
        )
        assert isinstance(result, TemperaturePrediction)

    def test_temp_in_reasonable_range(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        """Temperature should be in a reasonable range (0–50°C)."""
        energy = predict_energy(sample_reading, history_df, store)
        result = predict_temperature(
            sample_reading,
            history_df,
            energy.predicted_energy_kwh,
            store,
        )
        assert 0 < result.predicted_temp_c < 50


class TestCombinedPrediction:
    """Verify the chained prediction API."""

    def test_returns_combined_prediction(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        result = predict_combined(sample_reading, history_df, store)
        assert isinstance(result, CombinedPrediction)

    def test_combined_consistency(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
        store: ModelStore,
    ):
        """Combined result should match individual calls."""
        combined = predict_combined(sample_reading, history_df, store)
        energy = predict_energy(sample_reading, history_df, store)
        temp = predict_temperature(
            sample_reading,
            history_df,
            energy.predicted_energy_kwh,
            store,
        )
        assert abs(
            combined.predicted_energy_kwh - energy.predicted_energy_kwh
        ) < 1e-6
        assert abs(
            combined.predicted_temp_c - temp.predicted_temp_c
        ) < 1e-6


class TestNumericalCompatibility:
    """Compare core predictions against baseline values obtained from
    the trained models with the actual sample_test_data.csv.

    Baseline values were computed by running the models with their own
    ``feature_names_in_`` column ordering and the raw CSV data.
    """

    # Baselines from the inspect_models.py run:
    ENERGY_BASELINES = [
        128.8334, 125.9385, 121.1321, 120.5781, 119.0653,
        117.7931, 116.4715, 115.7844, 115.6968, 114.3872,
    ]
    TEMP_BASELINES = [
        32.1529, 31.8886, 31.5600, 31.7238, 31.5928,
        31.4248, 31.5194, 31.2377, 31.2729, 31.2024,
    ]

    def test_energy_baselines(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """First 10 rows should match baselines within tolerance."""
        for i in range(10):
            pred = predict_energy_from_row(sample_df.iloc[i], store)
            assert abs(pred - self.ENERGY_BASELINES[i]) < 0.01, (
                f"Row {i}: expected {self.ENERGY_BASELINES[i]:.4f}, "
                f"got {pred:.4f}"
            )

    def test_temp_baselines(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """Temperature predictions with chained energy should match."""
        for i in range(10):
            pred_e = predict_energy_from_row(sample_df.iloc[i], store)
            pred_t = predict_temp_from_row(
                sample_df.iloc[i], pred_e, store
            )
            assert abs(pred_t - self.TEMP_BASELINES[i]) < 0.01, (
                f"Row {i}: expected {self.TEMP_BASELINES[i]:.4f}, "
                f"got {pred_t:.4f}"
            )
