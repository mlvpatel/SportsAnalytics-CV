"""
Pydantic models for the SportsAnalytics-CV API.

Defines request/response schemas for all API endpoints.

Author: Malav Patel
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an analysis job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseModel):
    """Request body for video analysis."""

    video_path: str = Field(..., description="Path to the input video file")
    output_path: Optional[str] = Field(None, description="Path to save annotated output video")
    use_stubs: bool = Field(False, description="Use cached tracking data if available")
    export_stats: Optional[str] = Field(None, description="Path to export statistics JSON")


class PlayerStats(BaseModel):
    """Statistics for an individual player."""

    track_id: int
    team: int
    max_speed_kmh: float = 0.0
    total_distance_m: float = 0.0


class AnalysisResponse(BaseModel):
    """Response body with analysis results."""

    job_id: str
    status: JobStatus
    team1_possession: float = 0.0
    team2_possession: float = 0.0
    total_frames: int = 0
    processing_time_seconds: float = 0.0
    fps: float = 0.0
    player_stats: List[PlayerStats] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str = "healthy"
    version: str
    gpu_available: bool = False
    model_loaded: bool = False


class JobStatusResponse(BaseModel):
    """Response body for job status endpoint."""

    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    result: Optional[AnalysisResponse] = None
