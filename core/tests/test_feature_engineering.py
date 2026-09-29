# SPDX-License-Identifier: MIT
"""Tests for core.feature_engineering."""

import datetime

import numpy as np
import pandas as pd
import pytest

from core.config import ENERGY_MODEL_FEATURES, TEMP_MODEL_FEATURES
from core.feature_engineering import (
    FeatureEngineeringError,
    build_energy_features,
    build_energy_features_from_row,
    build_temp_features,
    build_temp_features_from_row,
    validate_feature_vector,
)
from core.model_loader import ModelStore
from core.schemas import SensorReading


class TestFeatureCount:
    """Verify feature vector dimensions."""

    def test_energy_features_46_columns(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        assert X.shape == (1, 46)

    def test_temp_features_46_columns(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_temp_features(sample_reading, history_df, 128.0)
        assert X.shape == (1, 46)


class TestFeatureOrdering:
    """Verify that the output column order exactly matches config."""

    def test_energy_column_order(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        assert list(X.columns) == ENERGY_MODEL_FEATURES

    def test_temp_column_order(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_temp_features(sample_reading, history_df, 128.0)
        assert list(X.columns) == TEMP_MODEL_FEATURES


class TestFeatureGeneration:
    """Verify individual feature categories are computed correctly."""

    def test_lag_features_present(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        lag_cols = [c for c in X.columns if "Lag" in c]
        assert len(lag_cols) == 16  # 4 sensors × 4 depths

    def test_rolling_features_present(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        rolling_cols = [c for c in X.columns if "Rolling" in c]
        assert len(rolling_cols) == 12

    def test_cyclical_features_present(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        cyclical_cols = [
            c for c in X.columns
            if c.endswith("_Sin") or c.endswith("_Cos")
        ]
        assert len(cyclical_cols) == 6

    def test_interaction_features_present(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        interaction_cols = [c for c in X.columns if "Interaction" in c]
        assert len(interaction_cols) == 4

    def test_cyclical_hour_sin_value(
        self,
        history_df: pd.DataFrame,
    ):
        """Test that Hour_Sin is correctly computed for hour=22."""
        reading = SensorReading(
            timestamp=datetime.datetime(2024, 8, 15, 22, 0, 0),
            chilled_water_rate=94.0,
            cooling_water_temp=32.4,
            building_load=505.9,
            outside_temp=82.0,
            dew_point=75.0,
            humidity=79.0,
            wind_speed=12.0,
            pressure=29.8,
            chiller_energy=132.0,
        )
        X = build_energy_features(reading, history_df)
        expected = np.sin(2 * np.pi * 22 / 24)
        assert abs(X.iloc[0]["Hour_Sin"] - expected) < 1e-10

    def test_interaction_load_temp(
        self,
        history_df: pd.DataFrame,
    ):
        reading = SensorReading(
            timestamp=datetime.datetime(2024, 8, 15, 22, 0, 0),
            chilled_water_rate=94.0,
            cooling_water_temp=32.4,
            building_load=505.9,
            outside_temp=82.0,
            dew_point=75.0,
            humidity=79.0,
            wind_speed=12.0,
            pressure=29.8,
            chiller_energy=132.0,
        )
        X = build_energy_features(reading, history_df)
        expected = 505.9 * 82.0
        assert abs(X.iloc[0]["Load_Temp_Interaction"] - expected) < 1e-6


class TestMissingFeatureValidation:
    """Verify that missing features raise errors, not silent zero-fill."""

    def test_insufficient_history_raises(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        """With only 5 rows of history, rolling-12 should fail."""
        short_history = history_df.head(5)
        with pytest.raises(FeatureEngineeringError, match="(Not enough|Need at least)"):
            build_energy_features(sample_reading, short_history)

    def test_missing_energy_column_in_history_raises(
        self,
        sample_reading: SensorReading,
        sample_df: pd.DataFrame,
    ):
        """History missing the 'Energy' column should raise."""
        bad_history = sample_df.drop(
            columns=["Energy_Lag_1"], errors="ignore"
        )
        # Remove any 'Energy' alias
        if "Energy" in bad_history.columns:
            bad_history = bad_history.drop(columns=["Energy"])
        with pytest.raises(FeatureEngineeringError, match="missing columns"):
            build_energy_features(sample_reading, bad_history)

    def test_no_nan_in_output(
        self,
        sample_reading: SensorReading,
        history_df: pd.DataFrame,
    ):
        X = build_energy_features(sample_reading, history_df)
        assert not X.isnull().any().any(), (
            f"NaN found in features: {X.columns[X.isnull().any()].tolist()}"
        )

    def test_validate_catches_wrong_column_count(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(FeatureEngineeringError, match="expected 46"):
            validate_feature_vector(df, ENERGY_MODEL_FEATURES)

    def test_validate_catches_nan(self):
        row = {col: 1.0 for col in ENERGY_MODEL_FEATURES}
        row[ENERGY_MODEL_FEATURES[5]] = float("nan")
        df = pd.DataFrame([row], columns=ENERGY_MODEL_FEATURES)
        with pytest.raises(FeatureEngineeringError, match="NaN"):
            validate_feature_vector(df, ENERGY_MODEL_FEATURES)


class TestPreEngineeredRowPath:
    """Tests for the from_row builders (simulation path)."""

    def test_energy_from_row_matches_model_features(
        self,
        sample_df: pd.DataFrame,
        store: ModelStore,
    ):
        row = sample_df.iloc[0]
        X = build_energy_features_from_row(row, store.energy_feature_names)
        assert list(X.columns) == store.energy_feature_names

    def test_temp_from_row_injects_energy(
        self,
        sample_df: pd.DataFrame,
        store: ModelStore,
    ):
        row = sample_df.iloc[0]
        pred_e = 128.83
        X = build_temp_features_from_row(
            row, store.temp_feature_names, pred_e
        )
        idx = store.temp_feature_names.index(
            "Chiller Energy Consumption (kWh)"
        )
        assert abs(X.iloc[0, idx] - pred_e) < 1e-10

    def test_temp_from_row_missing_col_raises(
        self,
        sample_df: pd.DataFrame,
        store: ModelStore,
    ):
        row = sample_df.iloc[0].drop("Humidity (%)")
        with pytest.raises(FeatureEngineeringError, match="missing"):
            build_temp_features_from_row(
                row, store.temp_feature_names, 128.0
            )
