# SPDX-License-Identifier: MIT
"""FastAPI application entrypoint for Predictive Cooling Optimizer backend."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import (
    data_router,
    health_router,
    models_router,
    optimization_router,
    predictions_router,
    simulation_router,
)
from backend.schemas.common import ErrorResponse
from core.feature_engineering import FeatureEngineeringError
from core.model_loader import ModelStore
from core.prediction import PredictionError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: preload models once on startup."""
    try:
        # Eagerly initialize and validate models into memory
        ModelStore.get()
    except Exception as exc:
        # Log or note error, health check will report degraded status
        pass
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Predictive Cooling Optimizer API",
        description=(
            "REST API providing temperature-aware predictive cooling optimization, "
            "canonical feature engineering, and deterministic simulation playback for data centers."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # CORS Configuration
    # -----------------------------------------------------------------------
    # Configurable through environment variable; defaults to common local frontend ports
    raw_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Exception Handlers
    # -----------------------------------------------------------------------
    @app.exception_handler(FeatureEngineeringError)
    async def feature_engineering_exception_handler(request: Request, exc: FeatureEngineeringError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=str(exc),
                error_type="FeatureEngineeringError",
                status_code=status.HTTP_400_BAD_REQUEST,
            ).model_dump(),
        )

    @app.exception_handler(PredictionError)
    async def prediction_exception_handler(request: Request, exc: PredictionError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail=str(exc),
                error_type="PredictionError",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=str(exc),
                error_type="ValueError",
                status_code=status.HTTP_400_BAD_REQUEST,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Format Pydantic validation errors cleanly without raw tracebacks
        errors = exc.errors()
        messages = []
        for err in errors:
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            messages.append(f"{loc}: {msg}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail="; ".join(messages),
                error_type="ValidationError",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                detail=str(exc.detail),
                error_type="HTTPException",
                status_code=exc.status_code,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        # Avoid exposing raw Python tracebacks
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="An internal server error occurred while processing the request.",
                error_type="InternalServerError",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ).model_dump(),
        )

    # -----------------------------------------------------------------------
    # Register Routers
    # -----------------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(predictions_router)
    app.include_router(optimization_router)
    app.include_router(simulation_router)
    app.include_router(data_router)

    return app


app = create_app()
