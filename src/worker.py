"""
Celery worker for SportsAnalytics-CV.

Handles asynchronous video analysis tasks using the SportsAnalyzer.

Author: Malav Patel
"""

import logging
import os
import sys

from celery import Celery

# Ensure the project root is in the Python path
sys.path.append(os.getcwd())

from main import SportsAnalyzer  # noqa: E402
from src.api.models import AnalysisResponse, JobStatus  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "sportsanalytics",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

# Configure Celery options
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(bind=True, name="analyze_video")
def analyze_video_task(
    self,
    video_path: str,
    output_path: str = None,
    use_stubs: bool = False,
    export_stats: str = None,
):
    """
    Celery task to run video analysis.

    Args:
        video_path: Path to input video.
        output_path: Path to output video (optional).
        use_stubs: Whether to use tracking stubs.
        export_stats: Path to export stats JSON (optional).

    Returns:
        Dict representation of AnalysisResponse.
    """
    job_id = self.request.id
    logger.info(f"Starting analysis task {job_id} for {video_path}")

    try:
        # Update state to PROCESSING
        self.update_state(
            state="STARTED", meta={"progress": 0.0, "message": "Initializing model..."}
        )

        # Initialize analyzer (this loads the model, which might take time)
        # Note: In a production worker, we might want to load the model globally
        # to avoid reloading it for every task, but for now we keep it safe.
        analyzer = SportsAnalyzer()

        self.update_state(
            state="STARTED", meta={"progress": 10.0, "message": "Running analysis..."}
        )

        # Run analysis
        result = analyzer.analyze(
            video_path=video_path,
            output_path=output_path,
            use_stubs=use_stubs,
            export_stats=export_stats,
        )

        # Map AnalysisResult (dataclass) to AnalysisResponse (Pydantic model)
        # We process player_stats mapping if needed
        api_player_stats = []
        if result.player_stats:
            # TODO: If player_stats structure changes in future, map it here.
            # Currently it is empty.
            pass

        response_model = AnalysisResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            team1_possession=result.team1_possession,
            team2_possession=result.team2_possession,
            total_frames=result.total_frames,
            processing_time_seconds=result.processing_time,
            fps=result.fps,
            player_stats=api_player_stats,
        )

        logger.info(f"Task {job_id} completed successfully")
        return response_model.model_dump()

    except Exception as e:
        logger.error(f"Task {job_id} failed: {e}")
        # celery will handle the failure state, but we log it.
        raise e
