"""
Video I/O utility functions for sports analytics.

Provides functions to read video files into frame arrays
and save processed frame arrays back to video files.

Author: Malav Patel
"""

import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_video(video_path: str) -> List[np.ndarray]:
    """
    Read a video file and return all frames as a list of numpy arrays.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of BGR frames as numpy arrays.

    Raises:
        FileNotFoundError: If the video file does not exist.
        IOError: If the video file cannot be opened or read.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video file: {video_path}. "
            "Check that the file exists and is a valid video format."
        )

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    if not frames:
        raise IOError(f"No frames could be read from video: {video_path}")

    logger.info(f"Read {frame_count} frames from {video_path}")
    return frames


def save_video(output_video_frames: List[np.ndarray], output_video_path: str) -> None:
    """
    Save a list of frames to a video file.

    Uses the XVID codec and writes at 24 FPS. The output
    resolution is determined by the first frame's dimensions.

    Args:
        output_video_frames: List of BGR frames as numpy arrays.
        output_video_path: Path where the output video will be saved.

    Raises:
        ValueError: If the frame list is empty.
        IOError: If the video writer cannot be initialized.
    """
    if not output_video_frames:
        raise ValueError("Cannot save video: frame list is empty")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    height, width = output_video_frames[0].shape[:2]

    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    if not out.isOpened():
        raise IOError(
            f"Cannot create video writer for: {output_video_path}. "
            "Check that the output directory exists and codec is available."
        )

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    logger.info(f"Saved {len(output_video_frames)} frames to {output_video_path}")
