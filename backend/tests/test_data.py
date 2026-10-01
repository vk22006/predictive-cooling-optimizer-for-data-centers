# SPDX-License-Identifier: MIT
"""Tests for sample data endpoint (/api/data/sample)."""

from fastapi.testclient import TestClient


def test_get_sample_data_default(client: TestClient):
    response = client.get("/api/data/sample")
    assert response.status_code == 200
    data = response.json()

    assert data["total_rows"] == 100
    assert data["offset"] == 0
    assert data["limit"] == 50
    assert len(data["rows"]) == 50
    assert "Chilled Water Rate (L/sec)" in data["columns"]


def test_get_sample_data_pagination(client: TestClient):
    response = client.get("/api/data/sample?offset=10&limit=5")
    assert response.status_code == 200
    data = response.json()

    assert data["offset"] == 10
    assert data["limit"] == 5
    assert len(data["rows"]) == 5


def test_get_sample_data_invalid_params(client: TestClient):
    res_neg = client.get("/api/data/sample?offset=-1")
    assert res_neg.status_code == 422

    res_limit = client.get("/api/data/sample?limit=0")
    assert res_limit.status_code == 422
