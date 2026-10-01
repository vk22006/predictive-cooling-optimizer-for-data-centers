# SPDX-License-Identifier: MIT
"""Prediction API endpoints for energy, temperature, and combined forecasting."""

from typing import Tuple
from fastapi import APIRouter
import pandas as pd

from backend.schemas.predictions import (
    CombinedPredictionRequest,
    CombinedPredictionResponse,
    EnergyPredictionRequest,
    EnergyPredictionResponse,
    TemperaturePredictionRequest,
    TemperaturePredictionResponse,
)
from core.model_loader import ModelStore
from core.prediction import (
    predict_combined,
    predict_energy,
    predict_energy_from_row,
    predict_temp_from_row,
    predict_temperature,
)
from core.schemas import SensorReading

router = APIRouter(prefix="/api/predict", tags=["predictions"])


def _extract_inputs(request) -> Tuple[bool, object, pd.DataFrame]:
    """Helper to convert API request into inputs required by the canonical core.

    Returns:
        (is_pre_engineered, reading_or_series, history_df)
    """
    if request.pre_engineered_row:
        return True, pd.Series(request.pre_engineered_row), pd.DataFrame()

    reading = SensorReading(
        timestamp=request.reading.timestamp,
        chilled_water_rate=request.reading.chilled_water_rate,
        cooling_water_temp=request.reading.cooling_water_temp,
        building_load=request.reading.building_load,
        outside_temp=request.reading.outside_temp,
        dew_point=request.reading.dew_point,
        humidity=request.reading.humidity,
        wind_speed=request.reading.wind_speed,
        pressure=request.reading.pressure,
        chiller_energy=request.reading.chiller_energy,
    )
    history_df = pd.DataFrame([h.model_dump(by_alias=True) for h in request.history])
    return False, reading, history_df


@router.post("/energy", response_model=EnergyPredictionResponse)
def post_predict_energy(request: EnergyPredictionRequest) -> EnergyPredictionResponse:
    """Predict chiller energy consumption (kWh) using the canonical core."""
    store = ModelStore.get()
    is_pre_eng, reading_obj, hist_df = _extract_inputs(request)

    if is_pre_eng:
        pred_kwh = predict_energy_from_row(reading_obj, store)
        ts = None
    else:
        result = predict_energy(reading_obj, hist_df, store)
        pred_kwh = result.predicted_energy_kwh
        ts = reading_obj.timestamp

    return EnergyPredictionResponse(
        predicted_energy_kwh=round(pred_kwh, 4),
        units="kWh",
        timestamp=ts,
        metadata={"model": "XGBRegressor", "target": "Chiller Energy Consumption (kWh)"},
    )


@router.post("/temperature", response_model=TemperaturePredictionResponse)
def post_predict_temperature(request: TemperaturePredictionRequest) -> TemperaturePredictionResponse:
    """Predict cooling water temperature (°C) 1 hour ahead using the canonical chained pipeline."""
    store = ModelStore.get()
    is_pre_eng, reading_obj, hist_df = _extract_inputs(request)

    if is_pre_eng:
        if request.predicted_energy_kwh is not None:
            energy_kwh = request.predicted_energy_kwh
        else:
            # Canonical chained evaluation: evaluate energy first
            energy_kwh = predict_energy_from_row(reading_obj, store)

        pred_temp = predict_temp_from_row(reading_obj, energy_kwh, store)
        ts = None
    else:
        if request.predicted_energy_kwh is not None:
            energy_kwh = request.predicted_energy_kwh
        else:
            # Canonical chained evaluation
            energy_res = predict_energy(reading_obj, hist_df, store)
            energy_kwh = energy_res.predicted_energy_kwh

        result = predict_temperature(reading_obj, hist_df, energy_kwh, store)
        pred_temp = result.predicted_temp_c
        ts = reading_obj.timestamp

    return TemperaturePredictionResponse(
        predicted_temp_c=round(pred_temp, 4),
        units="°C",
        chained_energy_kwh=round(energy_kwh, 4),
        timestamp=ts,
        metadata={
            "model": "XGBRegressor",
            "target": "Cooling Water Temperature (C)",
            "chained_dependency": "Chiller Energy Consumption (kWh)",
        },
    )


@router.post("/combined", response_model=CombinedPredictionResponse)
def post_predict_combined(request: CombinedPredictionRequest) -> CombinedPredictionResponse:
    """Run sequential chained inference to return both energy and temperature forecasts together."""
    store = ModelStore.get()
    is_pre_eng, reading_obj, hist_df = _extract_inputs(request)

    if is_pre_eng:
        pred_energy = predict_energy_from_row(reading_obj, store)
        pred_temp = predict_temp_from_row(reading_obj, pred_energy, store)
        ts = None
    else:
        combined = predict_combined(reading_obj, hist_df, store)
        pred_energy = combined.predicted_energy_kwh
        pred_temp = combined.predicted_temp_c
        ts = reading_obj.timestamp

    return CombinedPredictionResponse(
        predicted_energy_kwh=round(pred_energy, 4),
        predicted_temp_c=round(pred_temp, 4),
        energy_units="kWh",
        temperature_units="°C",
        timestamp=ts,
        metadata={"pipeline": "canonical_chained_v1"},
    )
