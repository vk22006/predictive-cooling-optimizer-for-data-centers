# SPDX-License-Identifier: MIT
"""Schemas for cooling setpoint optimization."""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.common import HistoricalDataPointSchema, SensorReadingSchema


class CandidatePointSchema(BaseModel):
    """Candidate chilled-water operating point evaluated during optimization."""
    chilled_water_rate_adjustment: float = Field(..., description="Additive adjustment applied in L/sec")
    chilled_water_rate: float = Field(..., description="Resulting candidate chilled water flow rate in L/sec")
    predicted_energy_kwh: float = Field(..., description="Model-predicted chiller energy consumption in kWh")
    predicted_temp_c: Optional[float] = Field(None, description="Model-predicted cooling water temperature in °C")
    is_selected: bool = Field(False, description="True if this candidate was selected as optimal")
    satisfies_temp_constraint: Optional[bool] = Field(
        None,
        description="Whether this candidate satisfies the requested upper temperature constraint"
    )


class OptimizationRequest(BaseModel):
    """Request schema for POST /api/optimize."""
    pre_engineered_row: Optional[Dict[str, float]] = Field(None, description="Pre-computed 46-feature row mapping")
    reading: Optional[SensorReadingSchema] = Field(None, description="Current raw sensor reading")
    history: Optional[List[HistoricalDataPointSchema]] = Field(None, description="Preceding time-series window (minimum 12 rows)")
    adjustments: Optional[List[float]] = Field(
        None,
        description="List of additive deltas (L/sec) to evaluate. Defaults to [-5, -2, -1, 0, 1, 2, 5]."
    )
    include_temp: bool = Field(True, description="Whether to compute temperature forecasts for each candidate")
    temperature_constraint_max: Optional[float] = Field(
        None,
        ge=10.0,
        le=50.0,
        description="Optional upper threshold for cooling water temperature in °C"
    )


class OptimizationResponse(BaseModel):
    """Response schema for cooling setpoint optimization."""
    baseline_energy_kwh: float = Field(..., description="Predicted energy consumption at current setpoint in kWh")
    optimized_energy_kwh: float = Field(..., description="Predicted energy consumption at optimal setpoint in kWh")
    estimated_energy_reduction_kwh: float = Field(
        ...,
        description="Model-estimated counterfactual energy reduction (baseline − optimized) in kWh"
    )
    savings_percent: float = Field(..., description="Estimated reduction as a percentage of baseline energy")
    current_chilled_water_rate: float = Field(..., description="Current operating chilled-water flow rate in L/sec")
    selected_chilled_water_rate: float = Field(..., description="Recommended optimal chilled-water flow rate in L/sec")
    selected_adjustment: float = Field(..., description="Selected additive adjustment in L/sec")
    baseline_temp_c: Optional[float] = Field(None, description="Predicted baseline cooling-water temperature in °C")
    suggestion: str = Field(..., description="Human-readable recommendation")
    candidates: List[CandidatePointSchema] = Field(default_factory=list, description="All evaluated candidate points")
    units: Dict[str, str] = Field(
        default_factory=lambda: {"energy": "kWh", "flow_rate": "L/sec", "temperature": "°C"},
        description="Units of measurement"
    )
    disclaimer: str = Field(
        "Values represent counterfactual model-estimated energy differences, not physically measured real-world savings.",
        description="Scientific qualification of optimization outputs"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "baseline_energy_kwh": 128.83,
                "optimized_energy_kwh": 128.60,
                "estimated_energy_reduction_kwh": 0.23,
                "savings_percent": 0.18,
                "current_chilled_water_rate": 94.0,
                "selected_chilled_water_rate": 89.0,
                "selected_adjustment": -5.0,
                "baseline_temp_c": 32.15,
                "suggestion": "Adjust chilled-water rate by -5.0 L/sec to save 0.23 kWh (0.2%).",
                "candidates": [
                    {
                        "chilled_water_rate_adjustment": -5.0,
                        "chilled_water_rate": 89.0,
                        "predicted_energy_kwh": 128.60,
                        "predicted_temp_c": 32.15,
                        "is_selected": True,
                        "satisfies_temp_constraint": True
                    }
                ],
                "units": {"energy": "kWh", "flow_rate": "L/sec", "temperature": "°C"},
                "disclaimer": "Values represent counterfactual model-estimated energy differences, not physically measured real-world savings."
            }
        }
    )
