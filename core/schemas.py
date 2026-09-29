# SPDX-License-Identifier: MIT
"""
Typed data structures for the canonical core.

Uses plain dataclasses (no FastAPI / Pydantic dependency) so that this
module can be consumed by any Python caller — Streamlit, FastAPI, CLI,
notebooks, or tests.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorReading:
    """A single timestamped set of raw sensor measurements.

    All values are in the units used during model training (see config.py
    for expected ranges).
    """
    timestamp: datetime.datetime
    chilled_water_rate: float          # L/sec
    cooling_water_temp: float          # °C
    building_load: float               # RT  (Refrigeration Tons)
    outside_temp: float                # °F
    dew_point: float                   # °F
    humidity: float                    # %
    wind_speed: float                  # mph
    pressure: float                    # inches Hg
    chiller_energy: float              # kWh  (current or most-recent reading)


# ---------------------------------------------------------------------------
# Prediction outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyPrediction:
    """Result of an energy-consumption prediction."""
    predicted_energy_kwh: float


@dataclass(frozen=True)
class TemperaturePrediction:
    """Result of a temperature forecast (cooling-water temp, 1 hr ahead)."""
    predicted_temp_c: float


@dataclass(frozen=True)
class CombinedPrediction:
    """Energy + temperature predictions returned together."""
    predicted_energy_kwh: float
    predicted_temp_c: float


# ---------------------------------------------------------------------------
# Optimization outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidatePoint:
    """One candidate operating point explored during optimization.

    Attributes:
        chilled_water_rate_adjustment: additive delta applied to the
            current chilled-water rate (L/sec).  Positive = increase.
        predicted_energy_kwh: model-predicted energy at this operating
            point.
        predicted_temp_c: model-predicted temperature at this operating
            point (if temp model is available).
        is_selected: True if this was the best candidate selected.
    """
    chilled_water_rate_adjustment: float
    predicted_energy_kwh: float
    predicted_temp_c: Optional[float] = None
    is_selected: bool = False


@dataclass(frozen=True)
class OptimizationResult:
    """Complete result from the cooling-optimization service.

    Attributes:
        baseline_energy_kwh: energy predicted at the current operating point.
        optimized_energy_kwh: energy predicted at the best operating point.
        energy_savings_kwh: baseline − optimized  (positive = savings).
        savings_percent: savings as a percentage of baseline.
        baseline_temp_c: temperature predicted at the current operating
            point (if temp model is available).
        suggestion: human-readable recommendation.
        candidates: all operating points evaluated during optimization.
    """
    baseline_energy_kwh: float
    optimized_energy_kwh: float
    energy_savings_kwh: float
    savings_percent: float
    baseline_temp_c: Optional[float] = None
    suggestion: str = ""
    candidates: List[CandidatePoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation schemas
# ---------------------------------------------------------------------------

@dataclass
class SimulationFrame:
    """One frame of output from a simulation step.

    Naming convention:  ``predicted_*`` = model output,
    ``lagged_*`` = proxy from the dataset (not ground truth).
    """
    step_index: int
    predicted_energy_kwh: float
    lagged_energy_kwh: float            # Energy_Lag_1 from the CSV row
    predicted_temp_c: float
    lagged_outside_temp_f: float        # OutsideTemp_Lag_1 from the CSV row
    potential_savings_pct: float         # (lagged − predicted) / lagged


@dataclass
class SimulationState:
    """Mutable state container for the live-simulation loop.

    The caller controls timing; the core advances one deterministic step
    per call to ``simulation_step()``.
    """
    current_index: int = 0
    is_running: bool = False
    history: List[SimulationFrame] = field(default_factory=list)
