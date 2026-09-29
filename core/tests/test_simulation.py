# SPDX-License-Identifier: MIT
"""Tests for core.simulation — deterministic one-step simulation."""

import pandas as pd
import pytest

from core.model_loader import ModelStore
from core.simulation import create_initial_state, simulation_step
from core.schemas import SimulationFrame, SimulationState


class TestSimulationCreation:
    """Verify initial state creation."""

    def test_initial_state(self, sample_df: pd.DataFrame):
        state = create_initial_state(sample_df)
        assert state.current_index == 0
        assert state.is_running is True
        assert state.history == []

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            create_initial_state(pd.DataFrame())


class TestSimulationStep:
    """Verify that simulation_step produces correct frames."""

    def test_first_step_returns_frame(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        next_state, frame = simulation_step(state, sample_df, store)
        assert frame is not None
        assert isinstance(frame, SimulationFrame)

    def test_step_advances_index(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        next_state, _ = simulation_step(state, sample_df, store)
        assert next_state.current_index == 1

    def test_frame_has_valid_energy(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        _, frame = simulation_step(state, sample_df, store)
        assert frame.predicted_energy_kwh > 0
        assert 30 < frame.predicted_energy_kwh < 300

    def test_frame_has_valid_temp(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        _, frame = simulation_step(state, sample_df, store)
        assert 0 < frame.predicted_temp_c < 50

    def test_frame_has_lagged_proxies(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        _, frame = simulation_step(state, sample_df, store)
        # Lagged energy should match Energy_Lag_1 from the CSV
        expected_lag = float(sample_df.iloc[0]["Energy_Lag_1"])
        assert abs(frame.lagged_energy_kwh - expected_lag) < 1e-6

    def test_multiple_steps(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        for i in range(5):
            state, frame = simulation_step(state, sample_df, store)
            assert frame is not None
            assert frame.step_index == i
        assert state.current_index == 5
        assert len(state.history) == 5

    def test_simulation_terminates(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """Run until the end of the dataset."""
        state = create_initial_state(sample_df)
        n = len(sample_df)
        for _ in range(n):
            state, frame = simulation_step(state, sample_df, store)
        assert state.is_running is False
        # One more step should return None
        state2, frame2 = simulation_step(state, sample_df, store)
        assert frame2 is None

    def test_history_accumulates(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        state = create_initial_state(sample_df)
        for _ in range(3):
            state, _ = simulation_step(state, sample_df, store)
        assert len(state.history) == 3

    def test_state_immutability(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """The original state should not be mutated."""
        state = create_initial_state(sample_df)
        original_idx = state.current_index
        next_state, _ = simulation_step(state, sample_df, store)
        assert state.current_index == original_idx  # unchanged
        assert next_state.current_index == original_idx + 1


class TestSimulationNumericalCompatibility:
    """Verify that simulation predictions match direct model calls."""

    def test_first_step_matches_baseline(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """Row 0 energy prediction from simulation should match the
        established baseline of 128.8334 kWh."""
        state = create_initial_state(sample_df)
        _, frame = simulation_step(state, sample_df, store)
        assert abs(frame.predicted_energy_kwh - 128.8334) < 0.01
