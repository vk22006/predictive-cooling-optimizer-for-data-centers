# SPDX-License-Identifier: MIT
"""Schemas for sample dataset inspection."""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class SampleDataResponse(BaseModel):
    """Paginated slice of pre-engineered historical test data."""
    total_rows: int = Field(..., description="Total rows in the sample dataset")
    offset: int = Field(..., ge=0, description="Starting row index")
    limit: int = Field(..., ge=1, description="Requested number of rows")
    columns: List[str] = Field(..., description="List of feature column names")
    rows: List[Dict[str, Any]] = Field(..., description="List of data records (row dictionary)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_rows": 100,
                "offset": 0,
                "limit": 2,
                "columns": ["Chilled Water Rate (L/sec)", "Cooling Water Temperature (C)", "..."],
                "rows": [
                    {
                        "Chilled Water Rate (L/sec)": 94.0,
                        "Cooling Water Temperature (C)": 32.4,
                        "Building Load (RT)": 505.9
                    }
                ]
            }
        }
    )
