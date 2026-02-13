"""
Configuration management for SportsAnalytics-CV.

Loads configuration from YAML file with environment variable
overrides and sensible defaults.

Author: Malav Patel
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


@dataclass
class ModelConfig:
    """Model configuration."""

    path: str = "models/best.pt"
    confidence: float = 0.5
    device: str = "cuda"


@dataclass
class TrackingConfig:
    """Tracking configuration."""

    max_player_ball_distance: float = 70.0
    ball_interpolation: bool = True
    detection_batch_size: int = 20


@dataclass
class SpeedConfig:
    """Speed estimation configuration."""

    frame_window: int = 5
    frame_rate: int = 24


@dataclass
class CameraConfig:
    """Camera movement estimation configuration."""

    minimum_distance: float = 5.0


@dataclass
class CourtConfig:
    """Court dimensions configuration."""

    width: float = 68.0
    length: float = 23.32


@dataclass
class OutputConfig:
    """Output configuration."""

    video_codec: str = "XVID"
    log_level: str = "INFO"
    log_file: str = "sportsanalytics.log"


@dataclass
class AppConfig:
    """Top-level application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load application configuration from YAML file.

    Supports environment variable overrides:
    - MODEL_PATH: Override model.path
    - DEVICE: Override model.device
    - CONFIDENCE: Override model.confidence
    - LOG_LEVEL: Override output.log_level

    Args:
        config_path: Path to YAML config file. If None, uses default.

    Returns:
        Populated AppConfig dataclass.
    """
    config = AppConfig()

    # Load from YAML if available
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}

            if "model" in data:
                config.model = ModelConfig(**data["model"])
            if "tracking" in data:
                config.tracking = TrackingConfig(**data["tracking"])
            if "speed" in data:
                config.speed = SpeedConfig(**data["speed"])
            if "camera" in data:
                config.camera = CameraConfig(**data["camera"])
            if "court" in data:
                config.court = CourtConfig(**data["court"])
            if "output" in data:
                config.output = OutputConfig(**data["output"])

            logger.info(f"Loaded config from {path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
    else:
        logger.info("No config file found, using defaults")

    # Environment variable overrides
    if os.environ.get("MODEL_PATH"):
        config.model.path = os.environ["MODEL_PATH"]
    if os.environ.get("DEVICE"):
        config.model.device = os.environ["DEVICE"]
    if os.environ.get("CONFIDENCE"):
        config.model.confidence = float(os.environ["CONFIDENCE"])
    if os.environ.get("LOG_LEVEL"):
        config.output.log_level = os.environ["LOG_LEVEL"]

    return config
