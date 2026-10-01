# Predictive Cooling Optimizer — Backend REST API Documentation

This document describes the FastAPI REST API providing machine learning inference, canonical feature engineering, cooling setpoint optimization, and deterministic simulation playback for data centers.

The backend is built with FastAPI, strictly typed with Pydantic v2 schemas, and serves as a thin API interface over the canonical computational core (`core/`).

---

## Getting Started & Local Development

### 1. Requirements & Environment
The backend requires Python 3.10+ and the packages listed in `backend/requirements.txt`:

```bash
pip install -r backend/requirements.txt
```

### 2. Starting the FastAPI Server
From the repository root directory, run `uvicorn`:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- **Swagger UI Interactive Documentation:** `http://127.0.0.1:8000/docs`
- **ReDoc Documentation:** `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON Specification:** `http://127.0.0.1:8000/openapi.json`

### 3. Environment Variables
- `CORS_ORIGINS`: Comma-separated list of allowed origin URLs for CORS (default: `"http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"`).

---

## Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Check service health, core availability, and model loading status |
| `GET` | `/api/models/info` | Discover model metadata, architectures, targets, and 46-feature schema |
| `POST` | `/api/predict/energy` | Forecast chiller energy consumption (kWh) |
| `POST` | `/api/predict/temperature` | Forecast cooling water temperature (°C) 1 hour ahead (chained inference) |
| `POST` | `/api/predict/combined` | Combined sequential energy and temperature predictions |
| `POST` | `/api/optimize` | Grid-search optimization for chilled-water setpoint recommendation |
| `POST` | `/api/simulation/step` | Deterministic one-step simulation advancement |
| `GET` | `/api/data/sample` | Paginated pre-engineered test dataset slice |

---

## API Reference

### 1. Health Check

#### `GET /api/health`
Verifies backend service readiness and ensures XGBoost models are loaded in memory.

##### Response Schema
```json
{
  "status": "string",
  "version": "string",
  "core_available": "boolean",
  "models_loaded": "boolean",
  "models": {
    "energy_model": "boolean",
    "temperature_model": "boolean"
  }
}
```

##### Example Response (200 OK)
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "core_available": true,
  "models_loaded": true,
  "models": {
    "energy_model": true,
    "temperature_model": true
  }
}
```

---

### 2. Model Metadata Discovery

#### `GET /api/models/info`
Returns sanitized metadata describing the deployed models, hyperparameters, and feature expectations without exposing internal filesystem paths.

##### Example Response (200 OK)
```json
{
  "energy_model": {
    "name": "Chiller Energy Prediction Model",
    "model_type": "XGBRegressor",
    "target_variable": "Chiller Energy Consumption (kWh)",
    "feature_count": 46,
    "features": [
      "Chilled Water Rate (L/sec)",
      "Cooling Water Temperature (C)",
      "Building Load (RT)",
      "Outside Temperature (F)",
      "Dew Point (F)",
      "Humidity (%)",
      "Wind Speed (mph)",
      "Pressure (in)",
      "Energy_Lag_1",
      "BuildingLoad_Lag_1",
      "OutsideTemp_Lag_1",
      "CoolingWaterTemp_Lag_1",
      "..."
    ],
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
    "features": [
      "Chilled Water Rate (L/sec)",
      "Building Load (RT)",
      "Chiller Energy Consumption (kWh)",
      "..."
    ],
    "hyperparameters": {
      "n_estimators": 150,
      "learning_rate": 0.1,
      "max_depth": 5
    }
  }
}
```

---

### 3. Energy Prediction

#### `POST /api/predict/energy`
Predicts chiller energy consumption in kWh. Accepts either raw sensor readings with a 12+ row historical window OR a pre-engineered 46-feature row.

##### Request Schema Options
1. **Raw Sensor Reading with History:**
```json
{
  "reading": {
    "timestamp": "2026-06-01T14:00:00",
    "chilled_water_rate": 95.0,
    "cooling_water_temp": 32.0,
    "building_load": 510.0,
    "outside_temp": 86.0,
    "dew_point": 75.0,
    "humidity": 78.0,
    "wind_speed": 7.0,
    "pressure": 29.82,
    "chiller_energy": 128.0
  },
  "history": [
    {
      "Energy": 120.0,
      "Building Load (RT)": 500.0,
      "Outside Temperature (F)": 84.0,
      "Cooling Water Temperature (C)": 31.5,
      "Chilled Water Rate (L/sec)": 94.0
    }
    // ... at least 12 historical entries
  ]
}
```

2. **Pre-engineered Feature Row (e.g. from sample dataset):**
```json
{
  "pre_engineered_row": {
    "Chilled Water Rate (L/sec)": 94.0,
    "Cooling Water Temperature (C)": 32.4,
    "Building Load (RT)": 505.9,
    "Outside Temperature (F)": 82.0,
    "Dew Point (F)": 74.0,
    "Humidity (%)": 76.0,
    "Wind Speed (mph)": 8.0,
    "Pressure (in)": 29.83,
    "Energy_Lag_1": 132.0,
    "BuildingLoad_Lag_1": 506.0,
    "OutsideTemp_Lag_1": 84.0,
    "CoolingWaterTemp_Lag_1": 32.0,
    "...": "all 46 columns"
  }
}
```

##### Example Response (200 OK)
```json
{
  "predicted_energy_kwh": 128.8334,
  "units": "kWh",
  "timestamp": "2026-06-01T14:00:00",
  "metadata": {
    "model": "XGBRegressor",
    "target": "Chiller Energy Consumption (kWh)"
  }
}
```

##### Error Responses
- `400 Bad Request`: When required features are missing or history has fewer than 12 entries (`FeatureEngineeringError`).
- `422 Unprocessable Content`: When request payload fails Pydantic schema validation.

---

### 4. Temperature Prediction (Chained Dependency)

#### `POST /api/predict/temperature`
Forecasts cooling water temperature 1 hour ahead (°C). If `predicted_energy_kwh` is omitted, the endpoint automatically evaluates the energy model first and injects its prediction into the temperature model's input vector.

##### Example Request
```json
{
  "pre_engineered_row": {
    "Chilled Water Rate (L/sec)": 94.0,
    "Cooling Water Temperature (C)": 32.4,
    "Building Load (RT)": 505.9,
    "...": "all 46 columns"
  }
}
```

##### Example Response (200 OK)
```json
{
  "predicted_temp_c": 32.1529,
  "units": "°C",
  "chained_energy_kwh": 128.8334,
  "timestamp": null,
  "metadata": {
    "model": "XGBRegressor",
    "target": "Cooling Water Temperature (C)",
    "chained_dependency": "Chiller Energy Consumption (kWh)"
  }
}
```

---

### 5. Combined Prediction

#### `POST /api/predict/combined`
Runs both energy and temperature predictions sequentially and returns unified results.

##### Example Response (200 OK)
```json
{
  "predicted_energy_kwh": 128.8334,
  "predicted_temp_c": 32.1529,
  "energy_units": "kWh",
  "temperature_units": "°C",
  "timestamp": null,
  "metadata": {
    "pipeline": "canonical_chained_v1"
  }
}
```

---

### 6. Cooling Setpoint Optimization

#### `POST /api/optimize`
Performs a grid search across discrete chilled-water flow rate adjustments (`[-5, -2, -1, 0, 1, 2, 5]` L/sec) bounded within physical training limits ($72.40$ to $141.50$ L/sec). Optionally checks if candidates satisfy a maximum temperature threshold.

##### Example Request
```json
{
  "pre_engineered_row": {
    "Chilled Water Rate (L/sec)": 94.0,
    "Cooling Water Temperature (C)": 32.4,
    "Building Load (RT)": 505.9,
    "...": "all 46 columns"
  },
  "adjustments": [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
  "include_temp": true,
  "temperature_constraint_max": 32.50
}
```

##### Example Response (200 OK)
```json
{
  "baseline_energy_kwh": 128.8334,
  "optimized_energy_kwh": 128.6019,
  "estimated_energy_reduction_kwh": 0.2315,
  "savings_percent": 0.18,
  "current_chilled_water_rate": 94.0,
  "selected_chilled_water_rate": 89.0,
  "selected_adjustment": -5.0,
  "baseline_temp_c": 32.1529,
  "suggestion": "Adjust chilled-water rate by -5.0 L/sec to save 0.23 kWh (0.2%).",
  "candidates": [
    {
      "chilled_water_rate_adjustment": -5.0,
      "chilled_water_rate": 89.0,
      "predicted_energy_kwh": 128.6019,
      "predicted_temp_c": 32.1529,
      "is_selected": true,
      "satisfies_temp_constraint": true
    },
    {
      "chilled_water_rate_adjustment": 0.0,
      "chilled_water_rate": 94.0,
      "predicted_energy_kwh": 128.8334,
      "predicted_temp_c": 32.1529,
      "is_selected": false,
      "satisfies_temp_constraint": true
    }
  ],
  "units": {
    "energy": "kWh",
    "flow_rate": "L/sec",
    "temperature": "°C"
  },
  "disclaimer": "Values represent counterfactual model-estimated energy differences, not physically measured real-world savings."
}
```

---

### 7. Simulation Playback

#### `POST /api/simulation/step`
Executes exactly one synchronous simulation frame. Never uses `time.sleep()`.

##### Example Request
```json
{
  "current_index": 0,
  "include_optimization": true
}
```

##### Example Response (200 OK)
```json
{
  "next_state": {
    "current_index": 1,
    "is_running": true,
    "total_steps": 100
  },
  "frame": {
    "step_index": 0,
    "predicted_energy_kwh": 128.8334,
    "lagged_energy_kwh": 132.0,
    "predicted_temp_c": 32.1529,
    "lagged_outside_temp_f": 84.0,
    "potential_savings_pct": 0.024,
    "current_chilled_water_rate": 94.0,
    "timestamp": "2024-07-01 00:00:00"
  },
  "optimization": {
    "baseline_energy_kwh": 128.8334,
    "optimized_energy_kwh": 128.6019,
    "estimated_energy_reduction_kwh": 0.2315,
    "savings_percent": 0.18,
    "selected_adjustment": -5.0
  }
}
```

---

### 8. Sample Dataset Inspection

#### `GET /api/data/sample?offset=0&limit=50`
Returns a paginated slice of test data for React dashboard charting.

##### Parameters
- `offset` (query, int): Starting record index (default `0`).
- `limit` (query, int): Number of records to return (default `50`, max `500`).

##### Example Response (200 OK)
```json
{
  "total_rows": 100,
  "offset": 0,
  "limit": 2,
  "columns": [
    "Chilled Water Rate (L/sec)",
    "Cooling Water Temperature (C)",
    "Building Load (RT)",
    "Outside Temperature (F)"
  ],
  "rows": [
    {
      "Chilled Water Rate (L/sec)": 94.0,
      "Cooling Water Temperature (C)": 32.4,
      "Building Load (RT)": 505.9,
      "Outside Temperature (F)": 82.0
    }
  ]
}
```

---

## Error Handling Standards

All endpoints return a uniform error structure in the event of client or processing failures:

```json
{
  "detail": "Descriptive human-readable error message",
  "error_type": "FeatureEngineeringError | ValidationError | HTTPException | InternalServerError",
  "status_code": 400
}
```

- **Stack Traces:** Raw Python tracebacks are suppressed in HTTP responses to prevent information leakage.
- **Validation Errors (422):** Formatted with field paths and specific violation reasons.
