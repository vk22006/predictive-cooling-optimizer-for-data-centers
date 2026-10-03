# SPDX-License-Identifier: MIT
"""Verification of the critical sequential prediction chain:

    input
      ↓
    energy prediction
      ↓
    predicted energy
      ↓
    temperature prediction

Verifies:
1. POST /api/predict/combined against canonical regression fixtures.
2. combined.energy == predict/energy.energy.
3. combined.temperature == predict/temperature using the chained energy prediction.
4. Automatic chained evaluation in POST /api/predict/temperature when predicted_energy_kwh is omitted.
5. Numerical consistency across both pre_engineered_row and raw reading paths.
"""

from typing import Any, Dict
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class TestPredictionChainIntegrity:
    """Verifies strict preservation of the sequential energy -> temperature pipeline."""

    def test_chained_parity_with_dataset_regression_fixtures(
        self,
        client: TestClient,
        regression_fixtures: Dict[str, Any],
        sample_test_df: pd.DataFrame,
    ):
        """Verify chain integrity for all canonical dataset regression fixture rows."""
        dataset_items = regression_fixtures["dataset_fixtures"]
        assert len(dataset_items) > 0

        for item in dataset_items:
            idx = item["sample_index"]
            expected_energy = item["expected_energy_kwh"]
            expected_temp = item["expected_temp_c"]
            row_dict = sample_test_df.iloc[idx].to_dict()

            # 1. Evaluate Combined Prediction
            res_comb = client.post("/api/predict/combined", json={"pre_engineered_row": row_dict})
            assert res_comb.status_code == 200
            comb_data = res_comb.json()
            comb_energy = comb_data["predicted_energy_kwh"]
            comb_temp = comb_data["predicted_temp_c"]

            # Combined predictions must match canonical regression fixture
            assert abs(comb_energy - expected_energy) < 1e-3, (
                f"Sample {idx}: combined energy {comb_energy} != expected {expected_energy}"
            )
            assert abs(comb_temp - expected_temp) < 1e-3, (
                f"Sample {idx}: combined temp {comb_temp} != expected {expected_temp}"
            )

            # 2. Evaluate Individual Energy Prediction
            res_energy = client.post("/api/predict/energy", json={"pre_engineered_row": row_dict})
            assert res_energy.status_code == 200
            pred_energy = res_energy.json()["predicted_energy_kwh"]

            # Energy endpoint must equal combined energy forecast
            assert abs(pred_energy - comb_energy) < 1e-4, (
                f"Sample {idx}: predict/energy ({pred_energy}) != combined.energy ({comb_energy})"
            )

            # 3. Evaluate Temperature Prediction with automatic chaining (omitted energy)
            res_temp_auto = client.post("/api/predict/temperature", json={"pre_engineered_row": row_dict})
            assert res_temp_auto.status_code == 200
            temp_auto_data = res_temp_auto.json()
            assert abs(temp_auto_data["predicted_temp_c"] - comb_temp) < 1e-4, (
                f"Sample {idx}: automatic predict/temp ({temp_auto_data['predicted_temp_c']}) != combined.temp ({comb_temp})"
            )
            assert abs(temp_auto_data["chained_energy_kwh"] - pred_energy) < 1e-4, (
                f"Sample {idx}: chained_energy_kwh ({temp_auto_data['chained_energy_kwh']}) != pred_energy ({pred_energy})"
            )

            # 4. Evaluate Temperature Prediction with explicitly injected chained energy
            res_temp_explicit = client.post(
                "/api/predict/temperature",
                json={"pre_engineered_row": row_dict, "predicted_energy_kwh": pred_energy},
            )
            assert res_temp_explicit.status_code == 200
            temp_explicit_data = res_temp_explicit.json()
            assert abs(temp_explicit_data["predicted_temp_c"] - comb_temp) < 1e-4, (
                f"Sample {idx}: explicit chained predict/temp ({temp_explicit_data['predicted_temp_c']}) != combined.temp ({comb_temp})"
            )
            assert abs(temp_explicit_data["chained_energy_kwh"] - pred_energy) < 1e-4

    def test_chained_parity_with_synthetic_reading_fixture(
        self,
        client: TestClient,
        regression_fixtures: Dict[str, Any],
    ):
        """Verify chain integrity for raw sensor reading + 13-row history pipeline."""
        item = regression_fixtures["synthetic_reading_fixture"]
        expected_energy = item["expected_energy_kwh"]
        expected_temp = item["expected_temp_c"]

        history_payload = [
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

        # 1. Combined Prediction
        res_comb = client.post("/api/predict/combined", json={"reading": reading_payload, "history": history_payload})
        assert res_comb.status_code == 200
        comb_data = res_comb.json()
        comb_energy = comb_data["predicted_energy_kwh"]
        comb_temp = comb_data["predicted_temp_c"]

        assert abs(comb_energy - expected_energy) < 1e-3
        assert abs(comb_temp - expected_temp) < 1e-3

        # 2. Individual Energy Prediction
        res_energy = client.post("/api/predict/energy", json={"reading": reading_payload, "history": history_payload})
        assert res_energy.status_code == 200
        pred_energy = res_energy.json()["predicted_energy_kwh"]
        assert abs(pred_energy - comb_energy) < 1e-4

        # 3. Individual Temperature Prediction with automatic chained energy evaluation
        res_temp_auto = client.post("/api/predict/temperature", json={"reading": reading_payload, "history": history_payload})
        assert res_temp_auto.status_code == 200
        temp_auto_data = res_temp_auto.json()
        assert abs(temp_auto_data["predicted_temp_c"] - comb_temp) < 1e-4
        assert abs(temp_auto_data["chained_energy_kwh"] - pred_energy) < 1e-4

        # 4. Individual Temperature Prediction with explicit chained energy
        res_temp_explicit = client.post(
            "/api/predict/temperature",
            json={"reading": reading_payload, "history": history_payload, "predicted_energy_kwh": pred_energy},
        )
        assert res_temp_explicit.status_code == 200
        temp_explicit_data = res_temp_explicit.json()
        assert abs(temp_explicit_data["predicted_temp_c"] - comb_temp) < 1e-4
        assert abs(temp_explicit_data["chained_energy_kwh"] - pred_energy) < 1e-4
