# SPDX-License-Identifier: MIT
"""Shared test fixtures and helpers for core tests."""

import datetime
import pandas as pd
import pytest

from core.config import SAMPLE_CSV
from core.model_loader import ModelStore
from core.schemas import SensorReading


@pytest.fixture(autouse=True)
def _reset_model_store():
    """Reset the ModelStore singleton between tests to prevent
    cross-test contamination."""
    ModelStore.reset()
    yield
    ModelStore.reset()


@pytest.fixture
def store() -> ModelStore:
    """Load the ModelStore singleton for tests that need it."""
    return ModelStore.get()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Load the sample_test_data.csv as a DataFrame."""
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def history_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Build a history DataFrame suitable for feature engineering.

    The sample CSV has pre-engineered columns, but the feature
    engineering pipeline needs columns named by their raw sensor
    names.  This fixture creates the necessary aliases.
    """
    df = sample_df.copy()
    # Alias 'Energy' from Energy_Lag_1 (matching Streamlit's approach)
    if "Energy" not in df.columns and "Energy_Lag_1" in df.columns:
        df["Energy"] = df["Energy_Lag_1"]
    # Ensure raw sensor columns exist
    required = [
        "Energy",
        "Building Load (RT)",
        "Outside Temperature (F)",
        "Cooling Water Temperature (C)",
    ]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(
                f"Test fixture: column '{col}' not in sample data"
            )
    return df


@pytest.fixture
def sample_reading(sample_df: pd.DataFrame) -> SensorReading:
    """Build a SensorReading from the first row of the sample data."""
    row = sample_df.iloc[0]
    return SensorReading(
        timestamp=datetime.datetime(2024, 8, 15, 22, 0, 0),
        chilled_water_rate=float(row["Chilled Water Rate (L/sec)"]),
        cooling_water_temp=float(row["Cooling Water Temperature (C)"]),
        building_load=float(row["Building Load (RT)"]),
        outside_temp=float(row["Outside Temperature (F)"]),
        dew_point=float(row["Dew Point (F)"]),
        humidity=float(row["Humidity (%)"]),
        wind_speed=float(row["Wind Speed (mph)"]),
        pressure=float(row["Pressure (in)"]),
        chiller_energy=float(row.get("Energy_Lag_1", 132.0)),
    )
