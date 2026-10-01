# SPDX-License-Identifier: MIT
"""Schemas for service health checks."""

from typing import Dict
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall service status, e.g. 'healthy'")
    version: str = Field(..., description="FastAPI application version")
    core_available: bool = Field(..., description="Whether the computational core package is available")
    models_loaded: bool = Field(..., description="Whether ML models are loaded and ready in memory")
    models: Dict[str, bool] = Field(..., description="Loading status of individual models")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "core_available": True,
                "models_loaded": True,
                "models": {
                    "energy_model": True,
                    "temperature_model": True
                }
            }
        }
    )
