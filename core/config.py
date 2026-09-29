# SPDX-License-Identifier: MIT
"""
Canonical configuration for the Predictive Cooling Optimizer core.

All paths, thresholds, and feature definitions live here so that every
other module in the package imports from one place.
"""

from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Paths — relative to the repository root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR: Path = _REPO_ROOT / "data"
MODELS_DIR: Path = _REPO_ROOT / "models"

SAMPLE_CSV: Path = DATA_DIR / "sample_test_data.csv"

ENERGY_MODEL_PKL: Path = MODELS_DIR / "energy_model.pkl"
TEMP_MODEL_PKL: Path = MODELS_DIR / "temp_model.pkl"
FEATURE_LIST_PKL: Path = MODELS_DIR / "feature_list.pkl"

# ---------------------------------------------------------------------------
# Optimization constraints (from notebook/optimization_engine.ipynb)
# ---------------------------------------------------------------------------
# These come from the training data's observed ranges.
TEMP_MAX: float = 28.0          # °C – upper safety limit
TEMP_MIN: float = 20.0          # °C – lower comfort limit
TEMP_TARGET: float = 24.0       # °C – ideal operating point

# Chilled-water rate range observed in training data.
CHILLED_WATER_MIN: float = 72.40  # L/sec
CHILLED_WATER_MAX: float = 141.50  # L/sec

# Cooling-water temperature range observed in training data.
COOLING_WATER_TEMP_MIN: float = 25.80  # °C
COOLING_WATER_TEMP_MAX: float = 36.20  # °C

# ---------------------------------------------------------------------------
# Minimum history required for feature engineering
# ---------------------------------------------------------------------------
# Rolling-12 requires 12 previous rows; total = 12 history + 1 current = 13.
MIN_HISTORY_ROWS: int = 13

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# These are the raw sensor / input columns that a user (or API caller)
# must provide before the feature-engineering pipeline can run.
RAW_INPUT_COLUMNS: List[str] = [
    "Chilled Water Rate (L/sec)",
    "Cooling Water Temperature (C)",
    "Building Load (RT)",
    "Outside Temperature (F)",
    "Dew Point (F)",
    "Humidity (%)",
    "Wind Speed (mph)",
    "Pressure (in)",
]

# Lag depths used by the feature-engineering pipeline.
LAG_DEPTHS: List[int] = [1, 2, 3, 6]

# Sensors for which lags are computed.
LAG_SENSORS: List[str] = [
    "Energy",               # alias of "Chiller Energy Consumption (kWh)"
    "Building Load (RT)",
    "Outside Temperature (F)",
    "Cooling Water Temperature (C)",
]

# Lag column naming convention: <SensorShort>_Lag_<depth>
LAG_SENSOR_SHORT_NAMES = {
    "Energy": "Energy",
    "Building Load (RT)": "BuildingLoad",
    "Outside Temperature (F)": "OutsideTemp",
    "Cooling Water Temperature (C)": "CoolingWaterTemp",
}

# Rolling window sizes.
ROLLING_WINDOWS: List[int] = [3, 6, 12]

# ---------------------------------------------------------------------------
# The energy model and temp model expect DIFFERENT feature orderings.
# These are extracted directly from each model's `feature_names_in_`
# attribute and verified against the pkl training artefacts.
#
# KEY DIFFERENCE:
#   - Energy model slot [1] = "Cooling Water Temperature (C)"
#   - Temp model   slot [2] = "Chiller Energy Consumption (kWh)"
#
# feature_list.pkl matches the TEMP model (not the energy model).
# ---------------------------------------------------------------------------

ENERGY_MODEL_FEATURES: List[str] = [
    "Chilled Water Rate (L/sec)",
    "Cooling Water Temperature (C)",       # <-- unique to energy model
    "Building Load (RT)",
    "Outside Temperature (F)",
    "Dew Point (F)",
    "Humidity (%)",
    "Wind Speed (mph)",
    "Pressure (in)",
    "Energy_Lag_1",
    "BuildingLoad_Lag_1",
    "OutsideTemp_Lag_1",
    "CoolingWaterTemp_Lag_1",
    "Energy_Lag_2",
    "BuildingLoad_Lag_2",
    "OutsideTemp_Lag_2",
    "CoolingWaterTemp_Lag_2",
    "Energy_Lag_3",
    "BuildingLoad_Lag_3",
    "OutsideTemp_Lag_3",
    "CoolingWaterTemp_Lag_3",
    "Energy_Lag_6",
    "BuildingLoad_Lag_6",
    "OutsideTemp_Lag_6",
    "CoolingWaterTemp_Lag_6",
    "Energy_RollingAvg_3",
    "BuildingLoad_RollingAvg_3",
    "OutsideTemp_RollingAvg_3",
    "Energy_RollingStd_3",
    "Energy_RollingAvg_6",
    "BuildingLoad_RollingAvg_6",
    "OutsideTemp_RollingAvg_6",
    "Energy_RollingStd_6",
    "Energy_RollingAvg_12",
    "BuildingLoad_RollingAvg_12",
    "OutsideTemp_RollingAvg_12",
    "Energy_RollingStd_12",
    "Hour_Sin",
    "Hour_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "Month_Sin",
    "Month_Cos",
    "Load_Temp_Interaction",
    "ChilledWater_Load_Interaction",
    "Temp_Humidity_Interaction",
    "Hour_Load_Interaction",
]

TEMP_MODEL_FEATURES: List[str] = [
    "Chilled Water Rate (L/sec)",
    "Building Load (RT)",
    "Chiller Energy Consumption (kWh)",    # <-- unique to temp model
    "Outside Temperature (F)",
    "Dew Point (F)",
    "Humidity (%)",
    "Wind Speed (mph)",
    "Pressure (in)",
    "Energy_Lag_1",
    "BuildingLoad_Lag_1",
    "OutsideTemp_Lag_1",
    "CoolingWaterTemp_Lag_1",
    "Energy_Lag_2",
    "BuildingLoad_Lag_2",
    "OutsideTemp_Lag_2",
    "CoolingWaterTemp_Lag_2",
    "Energy_Lag_3",
    "BuildingLoad_Lag_3",
    "OutsideTemp_Lag_3",
    "CoolingWaterTemp_Lag_3",
    "Energy_Lag_6",
    "BuildingLoad_Lag_6",
    "OutsideTemp_Lag_6",
    "CoolingWaterTemp_Lag_6",
    "Energy_RollingAvg_3",
    "BuildingLoad_RollingAvg_3",
    "OutsideTemp_RollingAvg_3",
    "Energy_RollingStd_3",
    "Energy_RollingAvg_6",
    "BuildingLoad_RollingAvg_6",
    "OutsideTemp_RollingAvg_6",
    "Energy_RollingStd_6",
    "Energy_RollingAvg_12",
    "BuildingLoad_RollingAvg_12",
    "OutsideTemp_RollingAvg_12",
    "Energy_RollingStd_12",
    "Hour_Sin",
    "Hour_Cos",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "Month_Sin",
    "Month_Cos",
    "Load_Temp_Interaction",
    "ChilledWater_Load_Interaction",
    "Temp_Humidity_Interaction",
    "Hour_Load_Interaction",
]
