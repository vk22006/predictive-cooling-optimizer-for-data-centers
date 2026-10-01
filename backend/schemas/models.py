# SPDX-License-Identifier: MIT
"""Schemas for model metadata and discovery."""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """Metadata describing a specific machine learning model."""
    name: str = Field(..., description="Descriptive name of the model")
    model_type: str = Field(..., description="Underlying model class / architecture")
    target_variable: str = Field(..., description="Predicted target variable and unit")
    feature_count: int = Field(..., description="Number of expected features")
    features: List[str] = Field(..., description="Ordered list of required feature names")
    hyperparameters: Dict[str, Any] = Field(..., description="Extracted model hyperparameters")


class ModelsInfoResponse(BaseModel):
    """Information regarding both models loaded in the backend."""
    energy_model: ModelInfo
    temperature_model: ModelInfo

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "energy_model": {
                    "name": "Chiller Energy Prediction Model",
                    "model_type": "XGBRegressor",
                    "target_variable": "Chiller Energy Consumption (kWh)",
                    "feature_count": 46,
                    "features": ["Chilled Water Rate (L/sec)", "Cooling Water Temperature (C)", "..."],
                    "hyperparameters": {
                        "n_estimators": 200,
                        "learning_rate": 0.1,
                        "max_depth": 6
                    }
                },
                "temperature_model": {
                    "name": "Cooling Water Temperature Forecasting Model (1-hr ahead)",
                    "model_type": "XGBRegressor",
                    "target_variable": "Cooling Water Temperature (C)",
                    "feature_count": 46,
                    "features": ["Chilled Water Rate (L/sec)", "Building Load (RT)", "..."],
                    "hyperparameters": {
                        "n_estimators": 150,
                        "learning_rate": 0.1,
                        "max_depth": 5
                    }
                }
            }
        }
    )
