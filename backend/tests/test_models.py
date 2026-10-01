# SPDX-License-Identifier: MIT
"""Tests for GET /api/models/info."""

from fastapi.testclient import TestClient


def test_models_info_success(client: TestClient):
    response = client.get("/api/models/info")
    assert response.status_code == 200
    data = response.json()

    assert "energy_model" in data
    assert "temperature_model" in data

    em = data["energy_model"]
    assert em["model_type"] == "XGBRegressor"
    assert em["target_variable"] == "Chiller Energy Consumption (kWh)"
    assert em["feature_count"] == 46
    assert len(em["features"]) == 46
    assert "n_estimators" in em["hyperparameters"]

    tm = data["temperature_model"]
    assert tm["model_type"] == "XGBRegressor"
    assert tm["target_variable"] == "Cooling Water Temperature (C)"
    assert tm["feature_count"] == 46
    assert len(tm["features"]) == 46
    assert "n_estimators" in tm["hyperparameters"]

    # Security check: no filesystem absolute paths exposed
    resp_text = response.text.lower()
    assert "c:\\" not in resp_text
    assert "/users/" not in resp_text
