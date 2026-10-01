# SPDX-License-Identifier: MIT
"""Data API endpoints for sample inspection."""

from fastapi import APIRouter, Query
from backend.schemas.data import SampleDataResponse
from backend.api.simulation import _get_simulation_df

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/sample", response_model=SampleDataResponse)
def get_sample_data(
    offset: int = Query(0, ge=0, description="Row offset for pagination"),
    limit: int = Query(50, ge=1, le=500, description="Max rows to return"),
) -> SampleDataResponse:
    """Return paginated pre-engineered sample dataset records for React dashboard visualizations."""
    df = _get_simulation_df()
    total_rows = len(df)
    sliced_df = df.iloc[offset : offset + limit]

    # Convert records to JSON-friendly dicts, replacing NaNs with None
    records = sliced_df.replace({float("nan"): None}).to_dict(orient="records")

    return SampleDataResponse(
        total_rows=total_rows,
        offset=offset,
        limit=limit,
        columns=list(df.columns),
        rows=records,
    )
