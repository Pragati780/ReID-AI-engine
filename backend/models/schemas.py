"""
backend/models/schemas.py

Pydantic models for the FastAPI layer. These describe the SHAPE of data
flowing over HTTP -- they do not perform any face-recognition logic
themselves. They exist purely so FastAPI can validate/document/serialize
requests and responses, and so the frontend has a stable, typed contract
to build against.

None of these models are imported by, or duplicate anything in, core/*.py.
The AI pipeline's own dataclasses (Appearance, MatchResult, etc.) stay
exactly where they are; this file only mirrors the shape of result.json
after the pipeline has already produced it.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class AppearanceSchema(BaseModel):
    """One continuous timestamp range where the reference person was found,
    mirroring core/aggregator.py's Appearance.to_dict() output exactly."""

    start_sec: float = Field(..., description="Appearance start time, in seconds")
    end_sec: float = Field(..., description="Appearance end time, in seconds")
    duration_sec: float = Field(..., description="end_sec - start_sec")
    max_similarity: float = Field(..., description="Highest cosine similarity observed in this appearance")
    avg_similarity: float = Field(..., description="Average cosine similarity across matched frames in this appearance")
    num_frames: int = Field(..., description="Number of sampled frames that matched within this appearance")


class ReferenceQualitySchema(BaseModel):
    """Quality report for the reference face, mirroring
    core/quality.py's QualityReport as saved into result.json."""

    sharpness_score: float
    width: int
    height: int
    is_low_quality: bool


class SearchResponse(BaseModel):
    """Response returned by POST /api/search and GET /api/result/{run_id}."""

    run_id: str
    processing_time_sec: Optional[float] = Field(
        None, description="Wall-clock time the pipeline took to run, in seconds. "
                           "May be null when re-fetched after a server restart."
    )
    person_found: bool
    num_appearances: int
    overall_best_similarity: Optional[float] = Field(
        None, description="Highest similarity score seen anywhere in the video, "
                           "or null if the person was never found."
    )
    reference_quality: ReferenceQualitySchema
    appearances: List[AppearanceSchema]
    matched_frames: List[str] = Field(default_factory=list, description="Filenames available via /api/frame/{run_id}/{filename}")
    annotated_frames: List[str] = Field(default_factory=list, description="Filenames available via /api/annotated/{run_id}/{filename}")
    reference_image_url: str = Field(..., description="URL to fetch the uploaded reference image back")
    csv_download_url: str
    json_download_url: str


class ErrorResponse(BaseModel):
    detail: str
