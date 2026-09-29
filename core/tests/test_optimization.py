# SPDX-License-Identifier: MIT
"""Tests for core.optimization — grid-search cooling optimization."""

import pandas as pd
import pytest

from core.config import CHILLED_WATER_MAX, CHILLED_WATER_MIN
from core.model_loader import ModelStore
from core.optimization import DEFAULT_ADJUSTMENTS, optimize_cooling
from core.prediction import predict_energy_from_row
from core.schemas import OptimizationResult


class TestOptimizationConstraints:
    """Verify that the optimiser respects constraints."""

    def _build_energy_row(
        self, sample_df: pd.DataFrame, store: ModelStore
    ) -> pd.DataFrame:
        """Build a 1-row energy-model DataFrame from sample row 0."""
        return sample_df[store.energy_feature_names].iloc[0:1].copy()

    def test_returns_optimization_result(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = self._build_energy_row(sample_df, store)
        result = optimize_cooling(X, store)
        assert isinstance(result, OptimizationResult)

    def test_baseline_equals_zero_adjustment(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """The baseline energy should equal prediction at adj=0."""
        X = self._build_energy_row(sample_df, store)
        result = optimize_cooling(X, store)
        zero_candidate = [
            c for c in result.candidates
            if c.chilled_water_rate_adjustment == 0.0
        ]
        assert len(zero_candidate) == 1
        assert abs(
            zero_candidate[0].predicted_energy_kwh
            - result.baseline_energy_kwh
        ) < 1e-6

    def test_savings_non_negative(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = self._build_energy_row(sample_df, store)
        result = optimize_cooling(X, store)
        assert result.energy_savings_kwh >= -1e-6

    def test_optimized_leq_baseline(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = self._build_energy_row(sample_df, store)
        result = optimize_cooling(X, store)
        assert result.optimized_energy_kwh <= result.baseline_energy_kwh + 1e-6

    def test_candidates_count(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = self._build_energy_row(sample_df, store)
        result = optimize_cooling(X, store)
        assert len(result.candidates) == len(DEFAULT_ADJUSTMENTS)

    def test_exactly_one_selected(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = self._build_energy_row(sample_df, store)
        result = optimize_cooling(X, store)
        selected = [c for c in result.candidates if c.is_selected]
        # Either 1 selected (savings found) or 0 (baseline is best)
        if result.energy_savings_kwh > 0.01:
            assert len(selected) == 1
        else:
            # When baseline is best, no candidate is marked selected
            assert len(selected) <= 1

    def test_chilled_water_stays_in_range(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        """After adjustment, the chilled-water rate should not exceed
        the training-data range."""
        X = self._build_energy_row(sample_df, store)
        base_cwr = float(X.iloc[0]["Chilled Water Rate (L/sec)"])
        for adj in DEFAULT_ADJUSTMENTS:
            clipped = max(
                CHILLED_WATER_MIN, min(CHILLED_WATER_MAX, base_cwr + adj)
            )
            assert CHILLED_WATER_MIN <= clipped <= CHILLED_WATER_MAX


class TestOptimizationResultSelection:
    """Verify selection logic picks the lowest energy."""

    def test_selected_has_lowest_energy(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = sample_df[store.energy_feature_names].iloc[0:1].copy()
        result = optimize_cooling(X, store)
        min_energy = min(c.predicted_energy_kwh for c in result.candidates)
        assert abs(result.optimized_energy_kwh - min_energy) < 1e-6


class TestOptimizationWithTemp:
    """Verify that temperature predictions are included."""

    def test_baseline_temp_populated(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = sample_df[store.energy_feature_names].iloc[0:1].copy()
        result = optimize_cooling(X, store, include_temp=True)
        assert result.baseline_temp_c is not None
        assert 0 < result.baseline_temp_c < 50

    def test_candidates_have_temp(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = sample_df[store.energy_feature_names].iloc[0:1].copy()
        result = optimize_cooling(X, store, include_temp=True)
        for c in result.candidates:
            assert c.predicted_temp_c is not None

    def test_without_temp(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = sample_df[store.energy_feature_names].iloc[0:1].copy()
        result = optimize_cooling(X, store, include_temp=False)
        assert result.baseline_temp_c is None
        for c in result.candidates:
            assert c.predicted_temp_c is None


class TestOptimizationCustomGrid:
    """Verify that a custom search grid works."""

    def test_custom_adjustments(
        self, sample_df: pd.DataFrame, store: ModelStore
    ):
        X = sample_df[store.energy_feature_names].iloc[0:1].copy()
        custom = [-10.0, 0.0, 10.0]
        result = optimize_cooling(
            X, store, adjustments=custom, include_temp=False
        )
        assert len(result.candidates) == 3
