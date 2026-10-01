# SPDX-License-Identifier: MIT
"""Simulation step API endpoint for dashboard simulation playback."""

from functools import lru_cache
from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd

from backend.schemas.optimization import CandidatePointSchema, OptimizationResponse
from backend.schemas.simulation import (
    SimulationFrameSchema,
    SimulationStateSchema,
    SimulationStepRequest,
    SimulationStepResponse,
)
from core.config import (
    CHILLED_WATER_MAX,
    CHILLED_WATER_MIN,
    ENERGY_MODEL_FEATURES,
    SAMPLE_CSV,
)
from core.model_loader import ModelStore
from core.optimization import optimize_cooling
from core.schemas import SimulationState
from core.simulation import simulation_step

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@lru_cache(maxsize=1)
def _get_simulation_df() -> pd.DataFrame:
    """Load and cache the pre-engineered simulation dataset."""
    if not SAMPLE_CSV.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Simulation dataset not found at expected location.",
        )
    return pd.read_csv(SAMPLE_CSV)


@router.post("/step", response_model=SimulationStepResponse)
def post_simulation_step(request: SimulationStepRequest) -> SimulationStepResponse:
    """Execute exactly one deterministic simulation step.

    Synchronous and stateless: advances from `request.current_index` without blocking or sleeping.
    """
    df = _get_simulation_df()
    total_steps = len(df)

    if request.current_index < 0 or request.current_index >= total_steps:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid step index {request.current_index}. Dataset contains {total_steps} steps (0 to {total_steps - 1}).",
        )

    store = ModelStore.get()
    state = SimulationState(current_index=request.current_index, is_running=True)

    next_state, frame = simulation_step(state, df, store)

    if frame is None:
        raise HTTPException(status_code=400, detail="Simulation reached end of dataset.")

    row = df.iloc[request.current_index]
    cwr = float(row.get("Chilled Water Rate (L/sec)", 0.0))

    frame_schema = SimulationFrameSchema(
        step_index=frame.step_index,
        predicted_energy_kwh=round(frame.predicted_energy_kwh, 4),
        lagged_energy_kwh=round(frame.lagged_energy_kwh, 4),
        predicted_temp_c=round(frame.predicted_temp_c, 4),
        lagged_outside_temp_f=round(frame.lagged_outside_temp_f, 4),
        potential_savings_pct=round(frame.potential_savings_pct, 4),
        current_chilled_water_rate=round(cwr, 2),
        timestamp=str(row.get("Local Time (Timezone : GMT+8h)", "")) if "Local Time (Timezone : GMT+8h)" in row else None,
    )

    state_schema = SimulationStateSchema(
        current_index=next_state.current_index,
        is_running=next_state.is_running,
        total_steps=total_steps,
    )

    opt_response = None
    if request.include_optimization:
        row_df = pd.DataFrame([row[ENERGY_MODEL_FEATURES].values], columns=ENERGY_MODEL_FEATURES)
        opt_res = optimize_cooling(row_df, store)

        candidates = [
            CandidatePointSchema(
                chilled_water_rate_adjustment=c.chilled_water_rate_adjustment,
                chilled_water_rate=round(float(np.clip(cwr + c.chilled_water_rate_adjustment, CHILLED_WATER_MIN, CHILLED_WATER_MAX)), 2),
                predicted_energy_kwh=round(c.predicted_energy_kwh, 4),
                predicted_temp_c=round(c.predicted_temp_c, 4) if c.predicted_temp_c is not None else None,
                is_selected=c.is_selected,
            )
            for c in opt_res.candidates
        ]

        selected_adj = next((c.chilled_water_rate_adjustment for c in opt_res.candidates if c.is_selected), 0.0)

        opt_response = OptimizationResponse(
            baseline_energy_kwh=round(opt_res.baseline_energy_kwh, 4),
            optimized_energy_kwh=round(opt_res.optimized_energy_kwh, 4),
            estimated_energy_reduction_kwh=round(opt_res.energy_savings_kwh, 4),
            savings_percent=round(opt_res.savings_percent, 2),
            current_chilled_water_rate=round(cwr, 2),
            selected_chilled_water_rate=round(float(np.clip(cwr + selected_adj, CHILLED_WATER_MIN, CHILLED_WATER_MAX)), 2),
            selected_adjustment=round(selected_adj, 2),
            baseline_temp_c=round(opt_res.baseline_temp_c, 4) if opt_res.baseline_temp_c is not None else None,
            suggestion=opt_res.suggestion,
            candidates=candidates,
            units={"energy": "kWh", "flow_rate": "L/sec", "temperature": "°C"},
            disclaimer="Values represent counterfactual model-estimated energy differences, not physically measured real-world savings.",
        )

    return SimulationStepResponse(
        next_state=state_schema,
        frame=frame_schema,
        optimization=opt_response,
    )
