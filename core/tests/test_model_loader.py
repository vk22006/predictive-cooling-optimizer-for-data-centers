# SPDX-License-Identifier: MIT
"""Tests for core.model_loader."""

import pytest
from core.config import ENERGY_MODEL_FEATURES, TEMP_MODEL_FEATURES
from core.model_loader import ModelLoadError, ModelStore


class TestModelLoading:
    """Verify that model artefacts load correctly and are validated."""

    def test_load_energy_model(self, store: ModelStore):
        assert store.energy_model is not None

    def test_load_temp_model(self, store: ModelStore):
        assert store.temp_model is not None

    def test_feature_list_pkl_loaded(self, store: ModelStore):
        assert len(store.feature_list_pkl) == 46

    def test_energy_feature_count(self, store: ModelStore):
        assert len(store.energy_feature_names) == 46

    def test_temp_feature_count(self, store: ModelStore):
        assert len(store.temp_feature_names) == 46

    def test_energy_features_match_config(self, store: ModelStore):
        assert store.energy_feature_names == ENERGY_MODEL_FEATURES

    def test_temp_features_match_config(self, store: ModelStore):
        assert store.temp_feature_names == TEMP_MODEL_FEATURES

    def test_energy_model_has_cooling_water_temp(self, store: ModelStore):
        """Energy model's differentiating feature."""
        assert "Cooling Water Temperature (C)" in store.energy_feature_names

    def test_temp_model_has_chiller_energy(self, store: ModelStore):
        """Temp model's differentiating feature."""
        assert "Chiller Energy Consumption (kWh)" in store.temp_feature_names

    def test_energy_model_does_not_have_chiller_energy(
        self, store: ModelStore
    ):
        assert (
            "Chiller Energy Consumption (kWh)"
            not in store.energy_feature_names
        )

    def test_temp_model_does_not_have_cooling_water_temp(
        self, store: ModelStore
    ):
        assert (
            "Cooling Water Temperature (C)" not in store.temp_feature_names
        )

    def test_singleton_returns_same_instance(self, store: ModelStore):
        store2 = ModelStore.get()
        assert store is store2

    def test_hyperparameters_populated(self, store: ModelStore):
        assert store.energy_n_estimators == 200
        assert store.energy_learning_rate == 0.1
        assert store.energy_max_depth == 6
        assert store.temp_n_estimators == 150
        assert store.temp_learning_rate == 0.1
        assert store.temp_max_depth == 5

    def test_feature_ordering_position(self, store: ModelStore):
        """Verify the critical positional difference."""
        assert store.energy_feature_names[1] == "Cooling Water Temperature (C)"
        assert store.temp_feature_names[2] == "Chiller Energy Consumption (kWh)"

    def test_load_with_invalid_path_raises(self):
        from pathlib import Path
        with pytest.raises(ModelLoadError, match="not found"):
            ModelStore._load(
                Path("nonexistent.pkl"),
                Path("nonexistent.pkl"),
                Path("nonexistent.pkl"),
            )
