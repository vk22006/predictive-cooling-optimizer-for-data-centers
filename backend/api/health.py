# SPDX-License-Identifier: MIT
"""Health check API endpoints."""

from fastapi import APIRouter
from backend.schemas.health import HealthResponse
from core.model_loader import ModelStore

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return backend health, core availability, and model loading status."""
    core_available = True
    models_loaded = False
    models_status = {"energy_model": False, "temperature_model": False}

    try:
        store = ModelStore.get()
        if store.energy_model is not None and store.temp_model is not None:
            models_loaded = True
            models_status["energy_model"] = True
            models_status["temperature_model"] = True
    except Exception:
        models_loaded = False

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        version="1.0.0",
        core_available=core_available,
        models_loaded=models_loaded,
        models=models_status,
    )
