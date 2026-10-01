# SPDX-License-Identifier: MIT
"""Tests for prediction endpoints (/api/predict/energy, /api/predict/temperature, /api/predict/combined)."""

from fastapi.testclient import TestClient


class TestEnergyPredictionAPI:
    def test_predict_energy_from_reading(self, client: TestClient, valid_sensor_reading, valid_history_list):
        payload = {
            "reading": valid_sensor_reading,
            "history": valid_history_list,
        }
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_energy_kwh" in data
        assert data["predicted_energy_kwh"] > 0
        assert data["units"] == "kWh"

    def test_predict_energy_from_row(self, client: TestClient, sample_row_0):
        payload = {"pre_engineered_row": sample_row_0}
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert abs(data["predicted_energy_kwh"] - 128.8334) < 1e-2

    def test_predict_energy_missing_inputs(self, client: TestClient):
        response = client.post("/api/predict/energy", json={})
        assert response.status_code == 422

    def test_predict_energy_insufficient_history(self, client: TestClient, valid_sensor_reading, valid_history_list):
        payload = {
            "reading": valid_sensor_reading,
            "history": valid_history_list[:5],  # only 5 rows, needs 12
        }
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code in (400, 422)


class TestTemperaturePredictionAPI:
    def test_predict_temperature_chained_from_row(self, client: TestClient, sample_row_0):
        # Omit predicted_energy_kwh to verify chained execution
        payload = {"pre_engineered_row": sample_row_0}
        response = client.post("/api/predict/temperature", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_temp_c" in data
        assert abs(data["predicted_temp_c"] - 32.1529) < 1e-2
        assert "chained_energy_kwh" in data
        assert abs(data["chained_energy_kwh"] - 128.8334) < 1e-2

    def test_predict_temperature_explicit_energy(self, client: TestClient, sample_row_0):
        payload = {
            "pre_engineered_row": sample_row_0,
            "predicted_energy_kwh": 130.0,
        }
        response = client.post("/api/predict/temperature", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["chained_energy_kwh"] == 130.0

    def test_predict_temperature_from_reading(self, client: TestClient, valid_sensor_reading, valid_history_list):
        payload = {
            "reading": valid_sensor_reading,
            "history": valid_history_list,
        }
        response = client.post("/api/predict/temperature", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_temp_c" in data
        assert 20.0 <= data["predicted_temp_c"] <= 45.0


class TestCombinedPredictionAPI:
    def test_predict_combined_from_row(self, client: TestClient, sample_row_0):
        payload = {"pre_engineered_row": sample_row_0}
        response = client.post("/api/predict/combined", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_energy_kwh" in data
        assert "predicted_temp_c" in data
        assert abs(data["predicted_energy_kwh"] - 128.8334) < 1e-2
        assert abs(data["predicted_temp_c"] - 32.1529) < 1e-2

    def test_predict_combined_from_reading(self, client: TestClient, valid_sensor_reading, valid_history_list):
        payload = {
            "reading": valid_sensor_reading,
            "history": valid_history_list,
        }
        response = client.post("/api/predict/combined", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_energy_kwh"] > 0
        assert data["predicted_temp_c"] > 0
