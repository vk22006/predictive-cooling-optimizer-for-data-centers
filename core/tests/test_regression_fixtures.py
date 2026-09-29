# SPDX-License-Identifier: MIT
"""Regression tests validating model and optimization outputs against deterministic fixtures."""

import datetime
import json
from pathlib import Path

import pandas as pd
import pytest

from core.config import ENERGY_MODEL_FEATURES, TEMP_MODEL_FEATURES
from core.model_loader import ModelStore
from core.optimization import optimize_cooling
from core.prediction import (
    predict_combined,
    predict_energy_from_row,
    predict_temp_from_row,
)
from core.schemas import SensorReading

FIXTURES_FILE = Path(__file__).parent / "fixtures" / "regression_fixtures.json"


@pytest.fixture(scope="module")
def fixture_data():
    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class TestDeterministicRegressionFixtures:
    """Verify that predictions and optimization outputs match fixed regression baselines."""

    def test_dataset_fixtures(self, fixture_data, sample_df: pd.DataFrame, store: ModelStore):
        for item in fixture_data["dataset_fixtures"]:
            idx = item["sample_index"]
            row = sample_df.iloc[idx]

            # 1. Energy prediction
            energy_pred = predict_energy_from_row(row, store)
            assert abs(energy_pred - item["expected_energy_kwh"]) < 1e-3, (
                f"Sample {idx} energy {energy_pred} != expected {item['expected_energy_kwh']}"
            )

            # 2. Temperature prediction (chained)
            temp_pred = predict_temp_from_row(row, energy_pred, store)
            assert abs(temp_pred - item["expected_temp_c"]) < 1e-3, (
                f"Sample {idx} temp {temp_pred} != expected {item['expected_temp_c']}"
            )

            # 3. Optimization
            row_df = pd.DataFrame([row[ENERGY_MODEL_FEATURES]])
            opt_result = optimize_cooling(row_df, store)

            assert abs(opt_result.baseline_energy_kwh - item["expected_opt_baseline_kwh"]) < 1e-3
            assert abs(opt_result.optimized_energy_kwh - item["expected_opt_optimized_kwh"]) < 1e-3
            assert abs(opt_result.energy_savings_kwh - item["expected_opt_savings_kwh"]) < 1e-3

            selected = [c for c in opt_result.candidates if c.is_selected]
            sel_adj = selected[0].chilled_water_rate_adjustment if selected else 0.0
            assert abs(sel_adj - item["expected_opt_adjustment"]) < 1e-5

    def test_synthetic_reading_fixture(self, fixture_data, store: ModelStore):
        item = fixture_data["synthetic_reading_fixture"]
        ts = datetime.datetime.fromisoformat(item["timestamp"])

        reading = SensorReading(
            timestamp=ts,
            chilled_water_rate=item["chilled_water_rate"],
            cooling_water_temp=item["cooling_water_temp"],
            building_load=item["building_load"],
            outside_temp=item["outside_temp"],
            dew_point=item["dew_point"],
            humidity=item["humidity"],
            wind_speed=item["wind_speed"],
            pressure=item["pressure"],
            chiller_energy=item["chiller_energy"],
        )

        history_df = pd.DataFrame({
            "Energy": [115.0, 118.0, 120.0, 122.0, 121.0, 125.0, 128.0, 130.0, 127.0, 124.0, 122.0, 126.0, 125.0],
            "Building Load (RT)": [480.0 + i * 5 for i in range(13)],
            "Outside Temperature (F)": [80.0 + (i % 4) for i in range(13)],
            "Cooling Water Temperature (C)": [31.0 + (i % 3) * 0.5 for i in range(13)],
            "Chilled Water Rate (L/sec)": [90.0 + (i % 5) for i in range(13)],
            "Dew Point (F)": [74.0] * 13,
            "Humidity (%)": [75.0] * 13,
            "Wind Speed (mph)": [5.0] * 13,
            "Pressure (in)": [29.8] * 13,
            "Chiller Energy Consumption (kWh)": [115.0, 118.0, 120.0, 122.0, 121.0, 125.0, 128.0, 130.0, 127.0, 124.0, 122.0, 126.0, 125.0],
        })

        pred = predict_combined(reading, history_df, store)
        assert abs(pred.predicted_energy_kwh - item["expected_energy_kwh"]) < 1e-3
        assert abs(pred.predicted_temp_c - item["expected_temp_c"]) < 1e-3
