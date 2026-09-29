# SPDX-License-Identifier: MIT
"""
Predictive Cooling Optimizer — Canonical Computational Core.

This package provides the canonical Python implementation for:
- Model loading and validation
- Feature engineering (46-feature pipeline)
- Energy and temperature prediction
- Cooling optimization (grid-search)
- Simulation state management
"""

from core.model_loader import ModelStore
from core.prediction import predict_energy, predict_temperature, predict_combined
from core.optimization import optimize_cooling
from core.simulation import SimulationState, simulation_step

__all__ = [
    "ModelStore",
    "predict_energy",
    "predict_temperature",
    "predict_combined",
    "optimize_cooling",
    "SimulationState",
    "simulation_step",
]
