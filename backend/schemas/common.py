# SPDX-License-Identifier: MIT
"""Common Pydantic models and schemas used across the API."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    detail: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Classification of the error")
    status_code: int = Field(..., description="HTTP status code")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Need at least 12 rows of history to compute rolling features (got 3).",
                "error_type": "FeatureEngineeringError",
                "status_code": 400
            }
        }
    )


class SensorReadingSchema(BaseModel):
    """Sensor reading input matching the canonical core.schemas.SensorReading."""
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of measurement")
    chilled_water_rate: float = Field(..., ge=0.0, le=500.0, description="Chilled water flow rate in L/sec")
    cooling_water_temp: float = Field(..., ge=0.0, le=80.0, description="Cooling water temperature in °C")
    building_load: float = Field(..., ge=0.0, le=5000.0, description="Building cooling load in RT")
    outside_temp: float = Field(..., ge=-40.0, le=150.0, description="Outside ambient temperature in °F")
    dew_point: float = Field(..., ge=-40.0, le=150.0, description="Dew point in °F")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity in %")
    wind_speed: float = Field(..., ge=0.0, le=150.0, description="Wind speed in mph")
    pressure: float = Field(..., ge=20.0, le=35.0, description="Atmospheric pressure in inches Hg")
    chiller_energy: Optional[float] = Field(None, ge=0.0, description="Current or recent chiller energy consumption in kWh")

    model_config = ConfigDict(
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "timestamp": "2026-06-01T14:00:00",
                "chilled_water_rate": 95.0,
                "cooling_water_temp": 32.0,
                "building_load": 510.0,
                "outside_temp": 86.0,
                "dew_point": 75.0,
                "humidity": 78.0,
                "wind_speed": 7.0,
                "pressure": 29.82,
                "chiller_energy": 128.0
            }
        }
    )


class HistoricalDataPointSchema(BaseModel):
    """One historical time-series observation required for lag and rolling calculations."""
    Energy: float = Field(..., ge=0.0, description="Chiller energy consumption in kWh")
    Building_Load_RT: float = Field(..., ge=0.0, alias="Building Load (RT)", description="Building cooling load in RT")
    Outside_Temperature_F: float = Field(..., ge=-40.0, le=150.0, alias="Outside Temperature (F)", description="Outside temperature in °F")
    Cooling_Water_Temperature_C: float = Field(..., ge=0.0, le=80.0, alias="Cooling Water Temperature (C)", description="Cooling water temperature in °C")
    Chilled_Water_Rate_L_sec: Optional[float] = Field(None, ge=0.0, le=500.0, alias="Chilled Water Rate (L/sec)", description="Chilled water rate in L/sec")
    Dew_Point_F: Optional[float] = Field(None, ge=-40.0, le=150.0, alias="Dew Point (F)", description="Dew point in °F")
    Humidity_Pct: Optional[float] = Field(None, ge=0.0, le=100.0, alias="Humidity (%)", description="Humidity %")
    Wind_Speed_mph: Optional[float] = Field(None, ge=0.0, le=150.0, alias="Wind Speed (mph)", description="Wind speed in mph")
    Pressure_in: Optional[float] = Field(None, ge=20.0, le=35.0, alias="Pressure (in)", description="Atmospheric pressure in inches Hg")
    Chiller_Energy_Consumption_kWh: Optional[float] = Field(None, ge=0.0, alias="Chiller Energy Consumption (kWh)", description="Chiller energy in kWh")

    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)
