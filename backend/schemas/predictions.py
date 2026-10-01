# SPDX-License-Identifier: MIT
"""Schemas for energy, temperature, and combined prediction endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.schemas.common import HistoricalDataPointSchema, SensorReadingSchema


class PredictionInputBase(BaseModel):
    """Base schema supporting either raw reading + history OR a pre-engineered feature row."""
    reading: Optional[SensorReadingSchema] = Field(None, description="Current raw sensor reading")
    history: Optional[List[HistoricalDataPointSchema]] = Field(None, description="Preceding time-series window (minimum 12 rows)")
    pre_engineered_row: Optional[Dict[str, float]] = Field(None, description="Pre-computed 46-feature row mapping")

    @model_validator(mode="after")
    def validate_inputs_present(self) -> "PredictionInputBase":
        has_reading = self.reading is not None
        has_pre_engineered = self.pre_engineered_row is not None and len(self.pre_engineered_row) > 0
        if not has_reading and not has_pre_engineered:
            raise ValueError("Must provide either 'reading' (with 'history') or 'pre_engineered_row'.")
        if has_reading and (self.history is None or len(self.history) < 12):
            raise ValueError("When providing 'reading', 'history' with at least 12 rows is required.")
        return self


class EnergyPredictionRequest(PredictionInputBase):
    """Request schema for POST /api/predict/energy."""
    pass


class EnergyPredictionResponse(BaseModel):
    """Response schema for energy prediction."""
    predicted_energy_kwh: float = Field(..., description="Forecasted chiller energy consumption in kWh")
    units: str = Field("kWh", description="Measurement unit")
    timestamp: Optional[datetime] = Field(None, description="Timestamp corresponding to the forecast")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and model metadata")


class TemperaturePredictionRequest(PredictionInputBase):
    """Request schema for POST /api/predict/temperature."""
    predicted_energy_kwh: Optional[float] = Field(
        None,
        description="Optional pre-computed chiller energy in kWh. If omitted, the energy model is evaluated first in a canonical chained sequence."
    )


class TemperaturePredictionResponse(BaseModel):
    """Response schema for temperature forecasting."""
    predicted_temp_c: float = Field(..., description="Forecasted cooling water temperature 1-hour ahead in °C")
    units: str = Field("°C", description="Measurement unit")
    chained_energy_kwh: Optional[float] = Field(None, description="Energy consumption value (kWh) injected into the temp model")
    timestamp: Optional[datetime] = Field(None, description="Timestamp corresponding to the forecast")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and model metadata")


class CombinedPredictionRequest(PredictionInputBase):
    """Request schema for POST /api/predict/combined."""
    pass


class CombinedPredictionResponse(BaseModel):
    """Response schema returning both energy and temperature forecasts together."""
    predicted_energy_kwh: float = Field(..., description="Forecasted chiller energy consumption in kWh")
    predicted_temp_c: float = Field(..., description="Forecasted cooling water temperature 1-hour ahead in °C")
    energy_units: str = Field("kWh", description="Energy unit")
    temperature_units: str = Field("°C", description="Temperature unit")
    timestamp: Optional[datetime] = Field(None, description="Timestamp corresponding to the forecast")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and model metadata")
