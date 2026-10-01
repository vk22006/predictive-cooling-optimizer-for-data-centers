# SPDX-License-Identifier: MIT
"""Backend API routers package."""

from backend.api.data import router as data_router
from backend.api.health import router as health_router
from backend.api.models import router as models_router
from backend.api.optimization import router as optimization_router
from backend.api.predictions import router as predictions_router
from backend.api.simulation import router as simulation_router

__all__ = [
    "health_router",
    "models_router",
    "predictions_router",
    "optimization_router",
    "simulation_router",
    "data_router",
]
