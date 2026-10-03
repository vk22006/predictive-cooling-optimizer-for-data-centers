# SPDX-License-Identifier: MIT
"""Comprehensive QA and hardening tests for malformed and invalid API requests.

Verifies:
- HTTP 422 for Pydantic schema validation failures
- HTTP 400 for semantically invalid application inputs
- No raw Python tracebacks or 500 errors exposed
- Physical constraints (negative values, out-of-range bounds, timestamps)
- Malformed pre-engineered rows and missing features
- Insufficient history windows
- NaN / Infinity injection attempts
"""

from datetime import datetime
import pytest
from fastapi.testclient import TestClient


class TestMalformedRequests:
    """Tests covering invalid and malformed request handling across all endpoints."""

    # -------------------------------------------------------------------------
    # 1. Missing request body
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/predict/energy",
            "/api/predict/temperature",
            "/api/predict/combined",
            "/api/optimize",
        ],
    )
    def test_missing_body_returns_422(self, client: TestClient, endpoint: str):
        """Endpoints requiring request bodies must reject requests without a body."""
        response = client.post(endpoint)
        assert response.status_code == 422
        data = response.json()
        assert data["error_type"] == "ValidationError"
        assert data["status_code"] == 422
        assert "detail" in data
        assert "Traceback" not in response.text

    # -------------------------------------------------------------------------
    # 2. Empty JSON object
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/predict/energy",
            "/api/predict/temperature",
            "/api/predict/combined",
            "/api/optimize",
        ],
    )
    def test_empty_json_returns_422(self, client: TestClient, endpoint: str):
        """Sending empty JSON {} must fail validation without raising 500 or tracebacks."""
        response = client.post(endpoint, json={})
        assert response.status_code == 422
        data = response.json()
        assert data["error_type"] == "ValidationError"
        assert data["status_code"] == 422
        assert "Traceback" not in response.text

    # -------------------------------------------------------------------------
    # 3. Missing required sensor fields
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "missing_field",
        [
            "chilled_water_rate",
            "cooling_water_temp",
            "building_load",
            "outside_temp",
            "dew_point",
            "humidity",
            "wind_speed",
            "pressure",
            "timestamp",
        ],
    )
    def test_missing_required_sensor_field(
        self, client: TestClient, valid_sensor_reading, valid_history_list, missing_field: str
    ):
        """Omitting any required sensor reading field must return 422."""
        bad_reading = valid_sensor_reading.copy()
        del bad_reading[missing_field]

        payload = {"reading": bad_reading, "history": valid_history_list}
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error_type"] == "ValidationError"
        assert missing_field in data["detail"]
        assert "Traceback" not in response.text

    # -------------------------------------------------------------------------
    # 4. Wrong field types
    # -------------------------------------------------------------------------
    def test_wrong_field_types(self, client: TestClient, valid_sensor_reading, valid_history_list):
        """Non-numeric or malformed data types must be cleanly rejected with 422."""
        bad_reading = valid_sensor_reading.copy()
        bad_reading["chilled_water_rate"] = "not_a_number"

        payload = {"reading": bad_reading, "history": valid_history_list}
        res = client.post("/api/predict/energy", json=payload)
        assert res.status_code == 422
        assert res.json()["error_type"] == "ValidationError"
        assert "Traceback" not in res.text

        # history must be a list
        res_hist = client.post("/api/predict/energy", json={"reading": valid_sensor_reading, "history": "not_a_list"})
        assert res_hist.status_code == 422
        assert res_hist.json()["error_type"] == "ValidationError"

        # current_index in simulation must be int
        res_sim = client.post("/api/simulation/step", json={"current_index": "not_an_int"})
        assert res_sim.status_code == 422
        assert res_sim.json()["error_type"] == "ValidationError"

    # -------------------------------------------------------------------------
    # 5. Null values in non-nullable fields
    # -------------------------------------------------------------------------
    def test_null_values_in_required_fields(self, client: TestClient, valid_sensor_reading, valid_history_list):
        """Null values in mandatory fields must return 422."""
        bad_reading = valid_sensor_reading.copy()
        bad_reading["chilled_water_rate"] = None

        payload = {"reading": bad_reading, "history": valid_history_list}
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error_type"] == "ValidationError"
        assert "Traceback" not in response.text

    def test_null_value_in_pre_engineered_row(self, client: TestClient, sample_row_0):
        """Null values inside pre_engineered_row dictionary must return 422."""
        bad_row = sample_row_0.copy()
        bad_row["Chilled Water Rate (L/sec)"] = None

        response = client.post("/api/predict/energy", json={"pre_engineered_row": bad_row})
        assert response.status_code == 422
        assert response.json()["error_type"] == "ValidationError"
        assert "Traceback" not in response.text

    # -------------------------------------------------------------------------
    # 6. NaN and Infinity injection attempts
    # -------------------------------------------------------------------------
    def test_nan_and_infinity_in_reading(self, client: TestClient):
        """NaN and Inf tokens in sensor reading must be rejected with 422."""
        raw_nan = b'{"reading": {"timestamp": "2026-06-01T14:00:00", "chilled_water_rate": NaN, "cooling_water_temp": 32.0, "building_load": 510.0, "outside_temp": 86.0, "dew_point": 75.0, "humidity": 78.0, "wind_speed": 7.0, "pressure": 29.82}, "history": []}'
        res_nan = client.post("/api/predict/energy", content=raw_nan, headers={"Content-Type": "application/json"})
        assert res_nan.status_code == 422
        assert res_nan.json()["error_type"] == "ValidationError"
        assert "Traceback" not in res_nan.text

        raw_inf = b'{"reading": {"timestamp": "2026-06-01T14:00:00", "chilled_water_rate": Infinity, "cooling_water_temp": 32.0, "building_load": 510.0, "outside_temp": 86.0, "dew_point": 75.0, "humidity": 78.0, "wind_speed": 7.0, "pressure": 29.82}, "history": []}'
        res_inf = client.post("/api/predict/energy", content=raw_inf, headers={"Content-Type": "application/json"})
        assert res_inf.status_code == 422
        assert res_inf.json()["error_type"] == "ValidationError"

    def test_nan_and_infinity_in_pre_engineered_row(self, client: TestClient):
        """NaN, Inf, and overflow floating-point numbers in pre_engineered_row must be rejected with 422."""
        raw_nan = b'{"pre_engineered_row": {"Chilled Water Rate (L/sec)": NaN}}'
        res_nan = client.post("/api/predict/energy", content=raw_nan, headers={"Content-Type": "application/json"})
        assert res_nan.status_code == 422
        assert res_nan.json()["error_type"] == "ValidationError"
        assert "Traceback" not in res_nan.text

        raw_inf = b'{"pre_engineered_row": {"Chilled Water Rate (L/sec)": Infinity}}'
        res_inf = client.post("/api/predict/energy", content=raw_inf, headers={"Content-Type": "application/json"})
        assert res_inf.status_code == 422
        assert res_inf.json()["error_type"] == "ValidationError"

        raw_overflow = b'{"pre_engineered_row": {"Chilled Water Rate (L/sec)": 1e999}}'
        res_over = client.post("/api/predict/energy", content=raw_overflow, headers={"Content-Type": "application/json"})
        assert res_over.status_code == 422
        assert res_over.json()["error_type"] == "ValidationError"

    # -------------------------------------------------------------------------
    # 7. Physical bounds and negative values where invalid
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "field,value",
        [
            ("chilled_water_rate", -1.0),
            ("cooling_water_temp", -0.5),
            ("building_load", -10.0),
            ("humidity", -1.0),
            ("humidity", 101.0),
            ("pressure", 19.0),
            ("pressure", 36.0),
            ("wind_speed", -5.0),
            ("outside_temp", -50.0),
            ("dew_point", -50.0),
        ],
    )
    def test_physical_bounds_violation_in_reading(
        self, client: TestClient, valid_sensor_reading, valid_history_list, field: str, value: float
    ):
        """Values exceeding physical domain constraints must return 422."""
        bad_reading = valid_sensor_reading.copy()
        bad_reading[field] = value

        payload = {"reading": bad_reading, "history": valid_history_list}
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code == 422
        assert response.json()["error_type"] == "ValidationError"
        assert field in response.json()["detail"]

    def test_negative_temperature_prediction_energy_override(self, client: TestClient, sample_row_0):
        """POST /api/predict/temperature must reject negative predicted_energy_kwh."""
        payload = {
            "pre_engineered_row": sample_row_0,
            "predicted_energy_kwh": -25.0,
        }
        response = client.post("/api/predict/temperature", json=payload)
        assert response.status_code == 422
        assert response.json()["error_type"] == "ValidationError"
        assert "greater than or equal to 0" in response.json()["detail"]

    def test_out_of_bounds_temperature_constraint_in_optimization(self, client: TestClient, sample_row_0):
        """POST /api/optimize must validate temperature_constraint_max within [10.0, 50.0]."""
        payload_low = {
            "pre_engineered_row": sample_row_0,
            "temperature_constraint_max": 5.0,  # Below 10.0
        }
        res_low = client.post("/api/optimize", json=payload_low)
        assert res_low.status_code == 422
        assert "temperature_constraint_max" in res_low.json()["detail"]

        payload_high = {
            "pre_engineered_row": sample_row_0,
            "temperature_constraint_max": 65.0,  # Above 50.0
        }
        res_high = client.post("/api/optimize", json=payload_high)
        assert res_high.status_code == 422
        assert "temperature_constraint_max" in res_high.json()["detail"]

    # -------------------------------------------------------------------------
    # 8. Impossible dates and timestamps
    # -------------------------------------------------------------------------
    def test_invalid_timestamps(self, client: TestClient, valid_sensor_reading, valid_history_list):
        """Invalid timestamp formats must be rejected with 422."""
        bad_reading = valid_sensor_reading.copy()
        bad_reading["timestamp"] = "not_a_date"

        payload = {"reading": bad_reading, "history": valid_history_list}
        response = client.post("/api/predict/energy", json=payload)
        assert response.status_code == 422
        assert response.json()["error_type"] == "ValidationError"
        assert "timestamp" in response.json()["detail"]

        bad_reading["timestamp"] = "2026-99-99T99:99:99"
        res_imp = client.post("/api/predict/energy", json={"reading": bad_reading, "history": valid_history_list})
        assert res_imp.status_code == 422
        assert res_imp.json()["error_type"] == "ValidationError"

    # -------------------------------------------------------------------------
    # 9. Malformed pre-engineered rows & unknown feature names
    # -------------------------------------------------------------------------
    def test_pre_engineered_row_missing_features(self, client: TestClient):
        """Pre-engineered row lacking required model features must return 400 Bad Request."""
        incomplete_row = {
            "Chilled Water Rate (L/sec)": 95.0,
            "Cooling Water Temperature (C)": 32.0,
        }
        response = client.post("/api/predict/energy", json={"pre_engineered_row": incomplete_row})
        assert response.status_code == 400
        data = response.json()
        assert data["error_type"] == "FeatureEngineeringError"
        assert "missing features" in data["detail"].lower()
        assert "Traceback" not in response.text

    def test_pre_engineered_row_only_unknown_features(self, client: TestClient):
        """Pre-engineered row containing only unrecognized feature names must return 400."""
        unknown_features_row = {
            "unrelated_sensor_a": 100.0,
            "unrelated_sensor_b": 200.0,
        }
        response = client.post("/api/predict/energy", json={"pre_engineered_row": unknown_features_row})
        assert response.status_code == 400
        data = response.json()
        assert data["error_type"] == "FeatureEngineeringError"
        assert "missing features" in data["detail"].lower()

    def test_pre_engineered_row_in_optimization_missing_features(self, client: TestClient):
        """POST /api/optimize with missing features in pre_engineered_row must return 400."""
        incomplete_row = {
            "Chilled Water Rate (L/sec)": 95.0,
        }
        response = client.post("/api/optimize", json={"pre_engineered_row": incomplete_row})
        assert response.status_code == 400
        data = response.json()
        assert data["error_type"] == "FeatureEngineeringError"
        assert "Missing required energy features" in data["detail"]

    # -------------------------------------------------------------------------
    # 10. Insufficient history for feature engineering
    # -------------------------------------------------------------------------
    def test_insufficient_history_in_predictions(self, client: TestClient, valid_sensor_reading, valid_history_list):
        """Providing fewer than 12 historical rows must fail with 422."""
        payload = {
            "reading": valid_sensor_reading,
            "history": valid_history_list[:11],  # 11 rows, minimum is 12
        }
        for endpoint in ["/api/predict/energy", "/api/predict/temperature", "/api/predict/combined", "/api/optimize"]:
            response = client.post(endpoint, json=payload)
            assert response.status_code == 422
            data = response.json()
            assert data["error_type"] == "ValidationError"
            assert "12 rows is required" in data["detail"]

    # -------------------------------------------------------------------------
    # 11. Pagination parameter validation in /api/data/sample
    # -------------------------------------------------------------------------
    def test_data_sample_invalid_parameters(self, client: TestClient):
        """GET /api/data/sample must reject invalid pagination queries."""
        # Negative offset
        res1 = client.get("/api/data/sample?offset=-5")
        assert res1.status_code == 422
        assert res1.json()["error_type"] == "ValidationError"

        # Limit zero
        res2 = client.get("/api/data/sample?limit=0")
        assert res2.status_code == 422

        # Limit exceeds max (500)
        res3 = client.get("/api/data/sample?limit=501")
        assert res3.status_code == 422

        # Non-integer query params
        res4 = client.get("/api/data/sample?limit=abc")
        assert res4.status_code == 422
