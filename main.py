"""
SportsAnalytics-CV: Real-Time Sports Analytics with Computer Vision

Main entry point for video analysis pipeline.

Author: Malav Patel
Email: malav.patel203@gmail.com
License: MIT
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.camera_movement import CameraMovementEstimator
from src.player_ball_assigner import PlayerBallAssigner
from src.speed_distance import SpeedDistanceEstimator
from src.team_assigner import TeamAssigner
from src.trackers import Tracker
from src.utils import read_video, save_video
from src.view_transformer import ViewTransformer

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
ALLOWED_EXPORT_EXTENSIONS = {".json"}
ALLOWED_MODEL_EXTENSIONS = {".pt", ".onnx"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB
MAX_EXPORT_PATH_LENGTH = 255
LOG_FILE = "sportsanalytics.log"

# ---------------------------------------------------------------------------
# Logging (sanitised formatter)
# ---------------------------------------------------------------------------


class SanitizedFormatter(logging.Formatter):
    """
    Logging formatter that strips control characters to prevent log injection.

    Newlines, carriage returns, and other control characters in user-supplied
    data could be used to forge log entries or break log parsers. This
    formatter replaces them with safe placeholders.
    """

    _CONTROL_RE = re.compile(r"[\r\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        message = super().format(record)
        return self._CONTROL_RE.sub("", message)


def _setup_logging() -> None:
    """Configure root logger with sanitised output."""
    fmt = SanitizedFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(fmt)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
    )


_setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_file_path(
    path: str,
    *,
    allowed_extensions: set[str],
    must_exist: bool = True,
    label: str = "File",
) -> Path:
    """
    Validate a file path for security and correctness.

    Checks:
    - Path is not empty or excessively long
    - Extension is in the allow-list
    - Resolved path does not escape project boundaries via symlinks
    - File exists (if must_exist=True) and is a regular file
    - File size is within limits

    Args:
        path: Raw path string from user input.
        allowed_extensions: Set of allowed lowercase extensions (e.g. {".mp4"}).
        must_exist: Whether the file must already exist on disk.
        label: Human-readable label for error messages.

    Returns:
        Resolved, validated Path object.

    Raises:
        ValueError: If any validation check fails.
    """
    if not path or not path.strip():
        raise ValueError(f"{label} path cannot be empty")

    if len(path) > MAX_EXPORT_PATH_LENGTH:
        raise ValueError(f"{label} path exceeds maximum length ({MAX_EXPORT_PATH_LENGTH})")

    # Reject null bytes (path-truncation attacks)
    if "\x00" in path:
        raise ValueError(f"{label} path contains null bytes")

    resolved = Path(path).resolve()

    # Extension check
    ext = resolved.suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(
            f"{label} has disallowed extension '{ext}'. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    if must_exist:
        # Symlink check – resolve() already follows symlinks, but we also
        # verify the final target is a regular file (not a device, FIFO, etc.)
        if not resolved.exists():
            raise FileNotFoundError(f"{label} not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"{label} is not a regular file: {resolved}")

        # File size guard
        file_size = resolved.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"{label} is too large ({file_size / 1e9:.1f} GB). "
                f"Maximum: {MAX_FILE_SIZE_BYTES / 1e9:.0f} GB"
            )
        if file_size == 0:
            raise ValueError(f"{label} is empty (0 bytes): {resolved}")
    else:
        # For output paths, ensure the parent directory exists
        parent = resolved.parent
        if not parent.exists():
            raise ValueError(f"Output directory does not exist: {parent}")

    return resolved


def _validate_confidence(value: float) -> float:
    """Validate confidence threshold is in (0, 1]."""
    if not 0.0 < value <= 1.0:
        raise ValueError(f"Confidence threshold must be in (0, 1], got {value}")
    return value


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    team1_possession: float
    team2_possession: float
    total_frames: int
    processing_time: float
    fps: float
    player_stats: dict

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "team1_possession": round(self.team1_possession, 2),
            "team2_possession": round(self.team2_possession, 2),
            "total_frames": self.total_frames,
            "processing_time": round(self.processing_time, 2),
            "fps": round(self.fps, 2),
            "player_stats": self.player_stats,
        }

    def to_json(self, filepath: str) -> None:
        """
        Export results to a JSON file.

        Args:
            filepath: Validated output path for the JSON file.
        """
        validated = _validate_file_path(
            filepath,
            allowed_extensions=ALLOWED_EXPORT_EXTENSIONS,
            must_exist=False,
            label="Export stats",
        )
        with open(validated, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------


class SportsAnalyzer:
    """
    Main class for sports video analysis.

    Performs object tracking, team classification, ball possession analysis,
    speed estimation, and generates annotated output videos.

    Example:
        >>> analyzer = SportsAnalyzer(model_path="models/best.pt")
        >>> result = analyzer.analyze("match.mp4", "output.mp4")
        >>> print(f"Team 1: {result.team1_possession}%")
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the sports analyzer.

        Args:
            model_path: Path to YOLO model weights.
            device: Computing device ('cuda' or 'cpu').
            conf_threshold: Detection confidence threshold in (0, 1].

        Raises:
            ValueError: If conf_threshold is out of range.
            FileNotFoundError: If model_path does not exist.
        """
        # Validate inputs
        validated_model = _validate_file_path(
            model_path,
            allowed_extensions=ALLOWED_MODEL_EXTENSIONS,
            must_exist=True,
            label="Model",
        )
        self.conf_threshold = _validate_confidence(conf_threshold)

        if device not in ("cuda", "cpu"):
            raise ValueError(f"Device must be 'cuda' or 'cpu', got '{device}'")

        self.model_path = str(validated_model)
        self.device = device

        logger.info("Initializing SportsAnalyzer")
        logger.info(f"Model: {validated_model.name}, Device: {device}")

        self.tracker = Tracker(self.model_path)

    def analyze(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        use_stubs: bool = False,
        stub_dir: str = "stubs",
        export_stats: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Analyze a video file.

        Args:
            video_path: Input video path (must be an allowed video format).
            output_path: Output video path (optional).
            use_stubs: Use cached tracking data.
            stub_dir: Directory for stub files.
            export_stats: Path to export statistics JSON.

        Returns:
            AnalysisResult object with statistics.

        Raises:
            FileNotFoundError: If the input video does not exist.
            ValueError: If any path validation fails.
        """
        # ---- Validate all paths up-front ----
        validated_input = _validate_file_path(
            video_path,
            allowed_extensions=ALLOWED_VIDEO_EXTENSIONS,
            must_exist=True,
            label="Input video",
        )

        validated_output = None
        if output_path:
            validated_output = _validate_file_path(
                output_path,
                allowed_extensions=ALLOWED_VIDEO_EXTENSIONS,
                must_exist=False,
                label="Output video",
            )

        if export_stats:
            _validate_file_path(
                export_stats,
                allowed_extensions=ALLOWED_EXPORT_EXTENSIONS,
                must_exist=False,
                label="Export stats",
            )

        start_time = time.time()
        logger.info(f"Starting analysis: {validated_input.name}")

        # Read video
        video_frames = read_video(str(validated_input))
        total_frames = len(video_frames)
        logger.info(f"Loaded {total_frames} frames")

        # Get object tracks
        stub_path = Path(stub_dir) / "track_stubs.pkl"
        tracks = self.tracker.get_object_tracks(
            video_frames,
            read_from_stub=use_stubs and stub_path.exists(),
            stub_path=str(stub_path) if use_stubs else None,
        )
        self.tracker.add_position_to_tracks(tracks)

        # Camera movement compensation
        camera_estimator = CameraMovementEstimator(video_frames[0])
        cam_stub = Path(stub_dir) / "camera_movement_stub.pkl"
        camera_movement = camera_estimator.get_camera_movement(
            video_frames,
            read_from_stub=use_stubs and cam_stub.exists(),
            stub_path=str(cam_stub) if use_stubs else None,
        )
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

        # View transformation
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)

        # Interpolate ball positions
        tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])

        # Speed and distance estimation
        speed_estimator = SpeedDistanceEstimator()
        speed_estimator.add_speed_and_distance_to_tracks(tracks)

        # Team assignment
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[
                    team
                ]

        # Ball possession analysis
        player_assigner = PlayerBallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

        team_ball_control = np.array(team_ball_control)

        # Calculate possession (with zero-division guard)
        total_control = len(team_ball_control)
        if total_control > 0:
            team1_possession = np.sum(team_ball_control == 1) / total_control * 100
            team2_possession = np.sum(team_ball_control == 2) / total_control * 100
        else:
            team1_possession = team2_possession = 0.0

        logger.info(
            f"Possession - Team 1: {team1_possession:.1f}%, " f"Team 2: {team2_possession:.1f}%"
        )

        # Generate output video
        if validated_output:
            logger.info("Generating annotated video...")
            output_frames = self.tracker.draw_annotations(video_frames, tracks, team_ball_control)
            output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movement)
            speed_estimator.draw_speed_and_distance(output_frames, tracks)
            save_video(output_frames, str(validated_output))
            logger.info(f"Output saved: {validated_output.name}")

        # Calculate processing metrics
        processing_time = time.time() - start_time
        fps = total_frames / processing_time if processing_time > 0 else 0.0

        result = AnalysisResult(
            team1_possession=team1_possession,
            team2_possession=team2_possession,
            total_frames=total_frames,
            processing_time=processing_time,
            fps=fps,
            player_stats={},
        )

        if export_stats:
            result.to_json(export_stats)
            logger.info("Statistics exported")

        logger.info(f"Analysis complete: {processing_time:.2f}s ({fps:.1f} FPS)")

        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point with argument parsing and validation."""
    parser = argparse.ArgumentParser(description="SportsAnalytics-CV: Real-Time Sports Analytics")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/input_videos/match.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/output_videos/output_video.avi",
        help="Output video path",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="models/best.pt",
        help="YOLO model path",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computing device",
    )
    parser.add_argument(
        "--use-stubs",
        action="store_true",
        help="Use cached tracking data",
    )
    parser.add_argument(
        "--export-stats",
        type=str,
        default=None,
        help="Export statistics to JSON",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (0-1)",
    )

    args = parser.parse_args()

    # Run analysis
    try:
        analyzer = SportsAnalyzer(
            model_path=args.model,
            device=args.device,
            conf_threshold=args.conf,
        )

        result = analyzer.analyze(
            video_path=args.input,
            output_path=args.output,
            use_stubs=args.use_stubs,
            export_stats=args.export_stats,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print("                 SPORTSANALYTICS-CV RESULTS")
    print("=" * 60)
    print(f"  Team 1 Possession: {result.team1_possession:.1f}%")
    print(f"  Team 2 Possession: {result.team2_possession:.1f}%")
    print(f"  Total Frames:      {result.total_frames}")
    print(f"  Processing Time:   {result.processing_time:.2f}s")
    print(f"  Average FPS:       {result.fps:.1f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
