# SPDX-License-Identifier: MIT
"""In-depth QA and constraint verification for the optimization endpoint (/api/optimize).

Verifies:
1. Candidate chilled-water flow rates stay strictly within [CHILLED_WATER_MIN, CHILLED_WATER_MAX].
2. Evaluated candidates belong to the configured optimization grid.
3. Baseline energy is consistent with the canonical core prediction.
4. Candidate predictions match the core optimizer directly.
5. Temperature predictions are populated when include_temp=True, and None when include_temp=False.
6. satisfies_temp_constraint is correctly computed without hard-filtering or dropping candidates.
7. Counterfactual disclaimer is returned on every response.
8. Soft temperature constraint reporting does NOT alter candidate selection logic.
"""

from typing import Dict
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.config import (
    CHILLED_WATER_MAX,
    CHILLED_WATER_MIN,
    ENERGY_MODEL_FEATURES,
)
from core.model_loader import ModelStore
from core.optimization import DEFAULT_ADJUSTMENTS, optimize_cooling


class TestOptimizationEndpointQA:
    """Rigorous QA tests for cooling setpoint optimization."""

    def test_candidate_flow_rates_within_training_bounds(self, client: TestClient, sample_row_0):
        """All candidates must remain within [CHILLED_WATER_MIN, CHILLED_WATER_MAX]."""
        payload = {"pre_engineered_row": sample_row_0, "include_temp": True}
        response = client.post("/api/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        for cand in data["candidates"]:
            flow_rate = cand["chilled_water_rate"]
            assert CHILLED_WATER_MIN <= flow_rate <= CHILLED_WATER_MAX, (
                f"Candidate flow rate {flow_rate} outside [{CHILLED_WATER_MIN}, {CHILLED_WATER_MAX}]"
            )

    def test_candidate_flow_rates_boundary_clipping(self, client: TestClient, sample_row_0):
        """Test boundary conditions near MIN and MAX to verify clipping."""
        # Row near minimum
        row_near_min = sample_row_0.copy()
        row_near_min["Chilled Water Rate (L/sec)"] = CHILLED_WATER_MIN + 1.0  # 73.4
        res_min = client.post(
            "/api/optimize",
            json={"pre_engineered_row": row_near_min, "adjustments": [-5.0, 0.0, 5.0]},
        )
        assert res_min.status_code == 200
        candidates_min = res_min.json()["candidates"]
        assert candidates_min[0]["chilled_water_rate"] == round(CHILLED_WATER_MIN, 2)  # Clipped

        # Row near maximum
        row_near_max = sample_row_0.copy()
        row_near_max["Chilled Water Rate (L/sec)"] = CHILLED_WATER_MAX - 1.0  # 140.5
        res_max = client.post(
            "/api/optimize",
            json={"pre_engineered_row": row_near_max, "adjustments": [-5.0, 0.0, 5.0]},
        )
        assert res_max.status_code == 200
        candidates_max = res_max.json()["candidates"]
        assert candidates_max[2]["chilled_water_rate"] == round(CHILLED_WATER_MAX, 2)  # Clipped

    def test_candidates_belong_to_configured_grid(self, client: TestClient, sample_row_0):
        """Candidate adjustments must exactly match requested adjustments."""
        # Default grid
        res_default = client.post("/api/optimize", json={"pre_engineered_row": sample_row_0})
        assert res_default.status_code == 200
        default_adjustments = [c["chilled_water_rate_adjustment"] for c in res_default.json()["candidates"]]
        assert default_adjustments == DEFAULT_ADJUSTMENTS

        # Custom grid
        custom_grid = [-4.0, -2.0, 0.0, 2.0, 4.0]
        res_custom = client.post(
            "/api/optimize",
            json={"pre_engineered_row": sample_row_0, "adjustments": custom_grid},
        )
        assert res_custom.status_code == 200
        custom_adjustments = [c["chilled_water_rate_adjustment"] for c in res_custom.json()["candidates"]]
        assert custom_adjustments == custom_grid

    def test_baseline_energy_consistency_with_core(self, client: TestClient, sample_row_0):
        """Baseline energy in API response must match core optimizer directly."""
        store = ModelStore.get()
        energy_df = pd.DataFrame([[sample_row_0[col] for col in ENERGY_MODEL_FEATURES]], columns=ENERGY_MODEL_FEATURES)
        core_opt = optimize_cooling(energy_df, store=store, include_temp=True)

        res = client.post("/api/optimize", json={"pre_engineered_row": sample_row_0, "include_temp": True})
        assert res.status_code == 200
        data = res.json()

        assert abs(data["baseline_energy_kwh"] - core_opt.baseline_energy_kwh) < 1e-4
        assert abs(data["optimized_energy_kwh"] - core_opt.optimized_energy_kwh) < 1e-4
        core_selected = next(c for c in core_opt.candidates if c.is_selected)
        assert abs(data["selected_adjustment"] - core_selected.chilled_water_rate_adjustment) < 1e-4

        # Candidate zero adjustment energy must match baseline energy
        cand_zero = next(c for c in data["candidates"] if c["chilled_water_rate_adjustment"] == 0.0)
        assert abs(cand_zero["predicted_energy_kwh"] - data["baseline_energy_kwh"]) < 1e-4

    def test_temperature_prediction_inclusion_and_omission(self, client: TestClient, sample_row_0):
        """Verifies predicted_temp_c is populated when requested and None when omitted."""
        # When include_temp=True
        res_with_temp = client.post("/api/optimize", json={"pre_engineered_row": sample_row_0, "include_temp": True})
        assert res_with_temp.status_code == 200
        data_with_temp = res_with_temp.json()
        assert data_with_temp["baseline_temp_c"] is not None
        for cand in data_with_temp["candidates"]:
            assert cand["predicted_temp_c"] is not None
            assert 20.0 <= cand["predicted_temp_c"] <= 50.0

        # When include_temp=False
        res_without_temp = client.post("/api/optimize", json={"pre_engineered_row": sample_row_0, "include_temp": False})
        assert res_without_temp.status_code == 200
        data_without_temp = res_without_temp.json()
        assert data_without_temp["baseline_temp_c"] is None
        for cand in data_without_temp["candidates"]:
            assert cand["predicted_temp_c"] is None

    def test_satisfies_temp_constraint_soft_reporting_no_filtering(self, client: TestClient, sample_row_0):
        """Constraint check must populate satisfies_temp_constraint without dropping or altering candidate selection."""
        # Unconstrained baseline
        res_base = client.post("/api/optimize", json={"pre_engineered_row": sample_row_0, "include_temp": True})
        assert res_base.status_code == 200
        base_candidates = res_base.json()["candidates"]
        assert len(base_candidates) == 7
        selected_cand_base = next(c for c in base_candidates if c["is_selected"])

        # Very high constraint (e.g. 45.0 °C) -> all should satisfy
        res_high = client.post(
            "/api/optimize",
            json={"pre_engineered_row": sample_row_0, "temperature_constraint_max": 45.0, "include_temp": True},
        )
        assert res_high.status_code == 200
        high_candidates = res_high.json()["candidates"]
        assert len(high_candidates) == 7
        assert all(c["satisfies_temp_constraint"] is True for c in high_candidates)
        # Selection remains identical (no hard filtering)
        selected_cand_high = next(c for c in high_candidates if c["is_selected"])
        assert selected_cand_high["chilled_water_rate_adjustment"] == selected_cand_base["chilled_water_rate_adjustment"]

        # Low constraint (e.g. 25.0 °C) -> candidates above 25.0 must report False
        res_low = client.post(
            "/api/optimize",
            json={"pre_engineered_row": sample_row_0, "temperature_constraint_max": 25.0, "include_temp": True},
        )
        assert res_low.status_code == 200
        low_candidates = res_low.json()["candidates"]
        assert len(low_candidates) == 7
        for c in low_candidates:
            if c["predicted_temp_c"] is not None:
                assert c["satisfies_temp_constraint"] == (c["predicted_temp_c"] <= 25.0)

        # Confirm candidate selection is NOT modified by the soft constraint
        selected_cand_low = next(c for c in low_candidates if c["is_selected"])
        assert selected_cand_low["chilled_water_rate_adjustment"] == selected_cand_base["chilled_water_rate_adjustment"]

    def test_disclaimer_present_and_accurate(self, client: TestClient, sample_row_0):
        """The API must explicitly disclaim counterfactual model estimation."""
        res = client.post("/api/optimize", json={"pre_engineered_row": sample_row_0})
        assert res.status_code == 200
        data = res.json()
        assert "disclaimer" in data
        assert "counterfactual" in data["disclaimer"].lower()
        assert "not physically measured" in data["disclaimer"].lower()
