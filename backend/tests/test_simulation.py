# SPDX-License-Identifier: MIT
"""Tests for simulation endpoint (/api/simulation/step)."""

from fastapi.testclient import TestClient


def test_simulation_step_0(client: TestClient):
    payload = {"current_index": 0, "include_optimization": False}
    response = client.post("/api/simulation/step", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["next_state"]["current_index"] == 1
    assert data["next_state"]["is_running"] is True
    assert data["next_state"]["total_steps"] == 100

    frame = data["frame"]
    assert frame["step_index"] == 0
    assert abs(frame["predicted_energy_kwh"] - 128.8334) < 1e-2
    assert abs(frame["predicted_temp_c"] - 32.1529) < 1e-2
    assert frame["lagged_energy_kwh"] == 132.0
    assert frame["lagged_outside_temp_f"] == 84.0
    assert data["optimization"] is None


def test_simulation_step_with_optimization(client: TestClient):
    payload = {"current_index": 0, "include_optimization": True}
    response = client.post("/api/simulation/step", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["optimization"] is not None
    assert data["optimization"]["baseline_energy_kwh"] > 0
    assert len(data["optimization"]["candidates"]) == 7


def test_simulation_step_invalid_index(client: TestClient):
    response = client.post("/api/simulation/step", json={"current_index": -1})
    assert response.status_code == 422  # ge=0 validation

    response_high = client.post("/api/simulation/step", json={"current_index": 9999})
    assert response_high.status_code == 400
    assert "Invalid step index" in response_high.json()["detail"]


def test_simulation_step_deterministic(client: TestClient):
    payload = {"current_index": 5, "include_optimization": False}
    res1 = client.post("/api/simulation/step", json=payload).json()
    res2 = client.post("/api/simulation/step", json=payload).json()
    assert res1 == res2
