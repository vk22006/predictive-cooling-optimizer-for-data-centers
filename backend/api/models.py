# SPDX-License-Identifier: MIT
"""Model information and metadata API endpoints."""

from fastapi import APIRouter
from backend.schemas.models import ModelInfo, ModelsInfoResponse
from core.model_loader import ModelStore

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/info", response_model=ModelsInfoResponse)
def get_models_info() -> ModelsInfoResponse:
    """Return sanitized metadata, feature schemas, and hyperparameters for deployed models."""
    store = ModelStore.get()

    energy_info = ModelInfo(
        name="Chiller Energy Prediction Model",
        model_type=type(store.energy_model).__name__,
        target_variable="Chiller Energy Consumption (kWh)",
        feature_count=len(store.energy_feature_names),
        features=store.energy_feature_names,
        hyperparameters={
            "n_estimators": store.energy_n_estimators,
            "learning_rate": store.energy_learning_rate,
            "max_depth": store.energy_max_depth,
        },
    )

    temp_info = ModelInfo(
        name="Cooling Water Temperature Forecasting Model (1-hr ahead)",
        model_type=type(store.temp_model).__name__,
        target_variable="Cooling Water Temperature (C)",
        feature_count=len(store.temp_feature_names),
        features=store.temp_feature_names,
        hyperparameters={
            "n_estimators": store.temp_n_estimators,
            "learning_rate": store.temp_learning_rate,
            "max_depth": store.temp_max_depth,
        },
    )

    return ModelsInfoResponse(
        energy_model=energy_info,
        temperature_model=temp_info,
    )
