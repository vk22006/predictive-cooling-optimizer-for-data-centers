# SPDX-License-Identifier: MIT
"""Tests for GET /api/health."""

from fastapi.testclient import TestClient


def test_health_success(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["core_available"] is True
    assert data["models_loaded"] is True
    assert data["models"]["energy_model"] is True
    assert data["models"]["temperature_model"] is True
