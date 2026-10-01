# SPDX-License-Identifier: MIT
"""Tests for optimization endpoint (/api/optimize)."""

from fastapi.testclient import TestClient


def test_optimize_from_row(client: TestClient, sample_row_0):
    payload = {
        "pre_engineered_row": sample_row_0,
        "include_temp": True,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "baseline_energy_kwh" in data
    assert "optimized_energy_kwh" in data
    assert "estimated_energy_reduction_kwh" in data
    assert "current_chilled_water_rate" in data
    assert "selected_chilled_water_rate" in data
    assert "candidates" in data
    assert len(data["candidates"]) == 7  # default grid size

    # Verify bounds
    for cand in data["candidates"]:
        assert 72.4 <= cand["chilled_water_rate"] <= 141.5
        assert cand["predicted_energy_kwh"] > 0
        assert cand["predicted_temp_c"] is not None

    # Verify baseline vs optimized
    assert data["optimized_energy_kwh"] <= data["baseline_energy_kwh"] + 1e-5


def test_optimize_with_temp_constraint(client: TestClient, sample_row_0):
    payload = {
        "pre_engineered_row": sample_row_0,
        "temperature_constraint_max": 32.20,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    for cand in data["candidates"]:
        if cand["predicted_temp_c"] is not None:
            expected_satisfies = cand["predicted_temp_c"] <= 32.20
            assert cand["satisfies_temp_constraint"] == expected_satisfies


def test_optimize_from_reading(client: TestClient, valid_sensor_reading, valid_history_list):
    payload = {
        "reading": valid_sensor_reading,
        "history": valid_history_list,
        "adjustments": [-2.0, 0.0, 2.0],
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["candidates"]) == 3


def test_optimize_missing_inputs(client: TestClient):
    response = client.post("/api/optimize", json={})
    assert response.status_code in (400, 422)
