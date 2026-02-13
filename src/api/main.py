"""
FastAPI application for SportsAnalytics-CV.

Provides REST API endpoints for video analysis, including
health checks, async job submission, and result retrieval.

Security features:
- Configurable CORS origins (default: restricted)
- Security headers middleware
- Optional API key authentication
- Per-IP rate limiting

Author: Malav Patel
"""

import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Dict, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

try:
    import torch

    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _GPU_AVAILABLE = False

from src.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
API_KEY: Optional[str] = os.environ.get("API_KEY")  # None = auth disabled
ALLOWED_ORIGINS: list[str] = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8501"
).split(",")
RATE_LIMIT_RPM: int = int(os.environ.get("RATE_LIMIT_RPM", "60"))

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SportsAnalytics-CV API",
    description="Real-time sports video analytics with computer vision",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Inject security headers on every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory, per-IP)
# ---------------------------------------------------------------------------
_rate_store: Dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Reject requests that exceed RATE_LIMIT_RPM per minute per IP."""
    if RATE_LIMIT_RPM <= 0:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0  # 1 minute

    # Prune old entries
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if now - t < window]

    if len(_rate_store[client_ip]) >= RATE_LIMIT_RPM:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Try again later."},
        )

    _rate_store[client_ip].append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# API key authentication (optional — only active when API_KEY env is set)
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(_api_key_header)):
    """Validate API key if authentication is enabled."""
    if API_KEY is None:
        return  # Auth disabled
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
jobs: Dict[str, JobStatusResponse] = {}


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------
def _run_analysis(job_id: str, request: AnalysisRequest) -> None:
    """
    Run video analysis as a background task.

    Args:
        job_id: Unique job identifier.
        request: Analysis request parameters.
    """
    try:
        jobs[job_id].status = JobStatus.PROCESSING
        jobs[job_id].message = "Analysis in progress"

        from main import SportsAnalyzer

        analyzer = SportsAnalyzer()
        result = analyzer.analyze(
            video_path=request.video_path,
            output_path=request.output_path,
            use_stubs=request.use_stubs,
            export_stats=request.export_stats,
        )

        jobs[job_id].status = JobStatus.COMPLETED
        jobs[job_id].progress = 100.0
        jobs[job_id].message = "Analysis completed successfully"
        jobs[job_id].result = AnalysisResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            team1_possession=result.team1_possession,
            team2_possession=result.team2_possession,
            total_frames=result.total_frames,
            processing_time_seconds=result.processing_time,
            fps=result.fps,
        )

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].message = str(e)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health, GPU availability, and model status."""
    return HealthResponse(
        status="healthy",
        version="1.1.0",
        gpu_available=_GPU_AVAILABLE,
        model_loaded=True,
    )


@app.post("/analyze", response_model=JobStatusResponse, dependencies=[Depends(verify_api_key)])
async def analyze_video(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    """
    Submit a video for analysis.

    Requires API key if API_KEY env var is set.
    """
    job_id = str(uuid.uuid4())

    job_status = JobStatusResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        message="Job submitted",
    )
    jobs[job_id] = job_status

    background_tasks.add_task(_run_analysis, job_id, request)
    logger.info(f"Job {job_id} submitted for {request.video_path}")

    return job_status


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return jobs[job_id]


@app.delete("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def delete_job(job_id: str) -> dict:
    """Delete a completed or failed job from memory."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if jobs[job_id].status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a job that is still processing",
        )

    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}
