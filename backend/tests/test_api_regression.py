# SPDX-License-Identifier: MIT
"""Regression tests verifying that API endpoints match canonical core fixtures without numerical drift."""

from fastapi.testclient import TestClient
import pandas as pd


def test_api_regression_dataset_fixtures(client: TestClient, regression_fixtures, sample_test_df: pd.DataFrame):
    """Test POST /api/predict and /api/optimize against fixed regression baselines."""
    for item in regression_fixtures["dataset_fixtures"]:
        idx = item["sample_index"]
        row_dict = sample_test_df.iloc[idx].to_dict()

        # 1. Energy prediction endpoint
        res_energy = client.post("/api/predict/energy", json={"pre_engineered_row": row_dict})
        assert res_energy.status_code == 200
        pred_energy = res_energy.json()["predicted_energy_kwh"]
        assert abs(pred_energy - item["expected_energy_kwh"]) < 1e-3, (
            f"Sample {idx} energy API drift: {pred_energy} vs expected {item['expected_energy_kwh']}"
        )

        # 2. Temperature prediction endpoint (chained dependency)
        res_temp = client.post("/api/predict/temperature", json={"pre_engineered_row": row_dict})
        assert res_temp.status_code == 200
        pred_temp = res_temp.json()["predicted_temp_c"]
        assert abs(pred_temp - item["expected_temp_c"]) < 1e-3, (
            f"Sample {idx} temp API drift: {pred_temp} vs expected {item['expected_temp_c']}"
        )

        # 3. Combined prediction endpoint
        res_combined = client.post("/api/predict/combined", json={"pre_engineered_row": row_dict})
        assert res_combined.status_code == 200
        comb_data = res_combined.json()
        assert abs(comb_data["predicted_energy_kwh"] - item["expected_energy_kwh"]) < 1e-3
        assert abs(comb_data["predicted_temp_c"] - item["expected_temp_c"]) < 1e-3

        # 4. Optimization endpoint
        res_opt = client.post("/api/optimize", json={"pre_engineered_row": row_dict, "include_temp": True})
        assert res_opt.status_code == 200
        opt_data = res_opt.json()
        assert abs(opt_data["baseline_energy_kwh"] - item["expected_opt_baseline_kwh"]) < 1e-3
        assert abs(opt_data["optimized_energy_kwh"] - item["expected_opt_optimized_kwh"]) < 1e-3
        assert abs(opt_data["estimated_energy_reduction_kwh"] - item["expected_opt_savings_kwh"]) < 1e-3
        assert abs(opt_data["selected_adjustment"] - item["expected_opt_adjustment"]) < 1e-3


def test_api_regression_synthetic_reading(client: TestClient, regression_fixtures, valid_history_list):
    """Test raw sensor reading through POST /api/predict/combined against regression fixture."""
    item = regression_fixtures["synthetic_reading_fixture"]

    # History defined in fixture
    fixed_history = [
        {
            "Energy": 115.0 if i not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) else [118.0, 120.0, 122.0, 121.0, 125.0, 128.0, 130.0, 127.0, 124.0, 122.0, 126.0, 125.0][i-1],
            "Building Load (RT)": 480.0 + i * 5,
            "Outside Temperature (F)": 80.0 + (i % 4),
            "Cooling Water Temperature (C)": 31.0 + (i % 3) * 0.5,
            "Chilled Water Rate (L/sec)": 90.0 + (i % 5),
            "Dew Point (F)": 74.0,
            "Humidity (%)": 75.0,
            "Wind Speed (mph)": 5.0,
            "Pressure (in)": 29.8,
            "Chiller Energy Consumption (kWh)": 115.0 if i not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) else [118.0, 120.0, 122.0, 121.0, 125.0, 128.0, 130.0, 127.0, 124.0, 122.0, 126.0, 125.0][i-1],
        }
        for i in range(13)
    ]

    reading_payload = {
        "timestamp": item["timestamp"],
        "chilled_water_rate": item["chilled_water_rate"],
        "cooling_water_temp": item["cooling_water_temp"],
        "building_load": item["building_load"],
        "outside_temp": item["outside_temp"],
        "dew_point": item["dew_point"],
        "humidity": item["humidity"],
        "wind_speed": item["wind_speed"],
        "pressure": item["pressure"],
        "chiller_energy": item["chiller_energy"],
    }

    res = client.post("/api/predict/combined", json={"reading": reading_payload, "history": fixed_history})
    assert res.status_code == 200
    data = res.json()
    assert abs(data["predicted_energy_kwh"] - item["expected_energy_kwh"]) < 1e-3
    assert abs(data["predicted_temp_c"] - item["expected_temp_c"]) < 1e-3
