# SPDX-License-Identifier: MIT
"""Pytest fixtures for FastAPI backend tests."""

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from backend.main import app
from core.config import SAMPLE_CSV
from core.model_loader import ModelStore

FIXTURES_FILE = Path(__file__).resolve().parent.parent.parent / "core" / "tests" / "fixtures" / "regression_fixtures.json"


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Create FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def regression_fixtures() -> Dict[str, Any]:
    """Load canonical regression fixtures."""
    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_test_df() -> pd.DataFrame:
    """Load sample test dataset."""
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def sample_row_0(sample_test_df: pd.DataFrame) -> Dict[str, float]:
    """Sample row 0 as a feature dictionary."""
    return sample_test_df.iloc[0].to_dict()


@pytest.fixture
def valid_sensor_reading() -> Dict[str, Any]:
    """A valid raw sensor reading payload."""
    return {
        "timestamp": "2026-06-01T14:00:00",
        "chilled_water_rate": 95.0,
        "cooling_water_temp": 32.0,
        "building_load": 510.0,
        "outside_temp": 86.0,
        "dew_point": 75.0,
        "humidity": 78.0,
        "wind_speed": 7.0,
        "pressure": 29.82,
        "chiller_energy": 128.0,
    }


@pytest.fixture
def valid_history_list() -> List[Dict[str, Any]]:
    """A valid 13-row historical sliding window."""
    return [
        {
            "Energy": 115.0 + i,
            "Building Load (RT)": 480.0 + i * 5,
            "Outside Temperature (F)": 80.0 + (i % 4),
            "Cooling Water Temperature (C)": 31.0 + (i % 3) * 0.5,
            "Chilled Water Rate (L/sec)": 90.0 + (i % 5),
            "Dew Point (F)": 74.0,
            "Humidity (%)": 75.0,
            "Wind Speed (mph)": 5.0,
            "Pressure (in)": 29.8,
            "Chiller Energy Consumption (kWh)": 115.0 + i,
        }
        for i in range(13)
    ]
