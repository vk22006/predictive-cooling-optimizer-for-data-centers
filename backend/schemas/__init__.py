# SPDX-License-Identifier: MIT
"""Backend Pydantic schemas package."""

from backend.schemas.common import (
    ErrorResponse,
    HistoricalDataPointSchema,
    SensorReadingSchema,
)
from backend.schemas.data import SampleDataResponse
from backend.schemas.health import HealthResponse
from backend.schemas.models import ModelInfo, ModelsInfoResponse
from backend.schemas.optimization import (
    CandidatePointSchema,
    OptimizationRequest,
    OptimizationResponse,
)
from backend.schemas.predictions import (
    CombinedPredictionRequest,
    CombinedPredictionResponse,
    EnergyPredictionRequest,
    EnergyPredictionResponse,
    TemperaturePredictionRequest,
    TemperaturePredictionResponse,
)
from backend.schemas.simulation import (
    SimulationFrameSchema,
    SimulationStateSchema,
    SimulationStepRequest,
    SimulationStepResponse,
)

__all__ = [
    "ErrorResponse",
    "SensorReadingSchema",
    "HistoricalDataPointSchema",
    "HealthResponse",
    "ModelInfo",
    "ModelsInfoResponse",
    "EnergyPredictionRequest",
    "EnergyPredictionResponse",
    "TemperaturePredictionRequest",
    "TemperaturePredictionResponse",
    "CombinedPredictionRequest",
    "CombinedPredictionResponse",
    "CandidatePointSchema",
    "OptimizationRequest",
    "OptimizationResponse",
    "SimulationStepRequest",
    "SimulationStateSchema",
    "SimulationFrameSchema",
    "SimulationStepResponse",
    "SampleDataResponse",
]
