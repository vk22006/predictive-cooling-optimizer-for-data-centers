# SPDX-License-Identifier: MIT
"""Schemas for simulation step execution."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.optimization import OptimizationResponse


class SimulationStepRequest(BaseModel):
    """Request schema for POST /api/simulation/step."""
    current_index: int = Field(0, ge=0, description="Zero-based index of the step to execute")
    include_optimization: bool = Field(False, description="Whether to also run setpoint optimization on this frame")

    model_config = ConfigDict(
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "current_index": 0,
                "include_optimization": False
            }
        }
    )


class SimulationStateSchema(BaseModel):
    """State of the live simulation loop."""
    current_index: int = Field(..., description="Index of the just-executed step")
    is_running: bool = Field(..., description="Whether further steps remain in the dataset")
    total_steps: int = Field(..., description="Total available steps in the simulation dataset")


class SimulationFrameSchema(BaseModel):
    """Output metrics for a single simulation frame."""
    step_index: int = Field(..., description="Step index in the sequence")
    predicted_energy_kwh: float = Field(..., description="Forecasted chiller energy consumption in kWh")
    lagged_energy_kwh: float = Field(..., description="Historical energy consumption proxy from 1 hour ago")
    predicted_temp_c: float = Field(..., description="Forecasted cooling water temperature in °C")
    lagged_outside_temp_f: float = Field(..., description="Historical outside temperature proxy from 1 hour ago in °F")
    potential_savings_pct: float = Field(..., description="Relative percentage difference between lagged and predicted energy")
    current_chilled_water_rate: float = Field(..., description="Chilled water flow rate in L/sec for this step")
    timestamp: Optional[str] = Field(None, description="ISO timestamp if available in dataset")


class SimulationStepResponse(BaseModel):
    """Response returned by POST /api/simulation/step."""
    next_state: SimulationStateSchema
    frame: SimulationFrameSchema
    optimization: Optional[OptimizationResponse] = Field(
        None,
        description="Optional optimization evaluation if 'include_optimization' was set to True"
    )
