"""
Speed and distance estimation module for sports analytics.

Calculates player speed (km/h) and distance covered (meters)
using transformed court positions across frame windows.

Author: Malav Patel
"""

import logging
from typing import Dict, List

import cv2
import numpy as np

from src.utils import get_foot_position, measure_distance

logger = logging.getLogger(__name__)


class SpeedDistanceEstimator:
    """
    Estimates speed and total distance for tracked players.

    Uses a sliding window approach over frames to calculate
    instantaneous speed and cumulative distance. Positions must
    be pre-transformed to real-world court coordinates.

    Attributes:
        frame_window: Number of frames per speed calculation window.
        frame_rate: Video frame rate in FPS.
    """

    def __init__(self, frame_window: int = 5, frame_rate: int = 24) -> None:
        """
        Initialize the speed and distance estimator.

        Args:
            frame_window: Number of frames to average speed over.
                Larger values produce smoother but less responsive estimates.
            frame_rate: Video frame rate in frames per second.
        """
        if frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {frame_rate}")
        if frame_window <= 0:
            raise ValueError(f"frame_window must be positive, got {frame_window}")

        self.frame_window = frame_window
        self.frame_rate = frame_rate

    def add_speed_and_distance_to_tracks(self, tracks: Dict[str, List[Dict[int, dict]]]) -> None:
        """
        Calculate and add speed/distance data to all player tracks.

        Modifies the tracks dictionary in-place, adding 'speed' (km/h)
        and 'distance' (meters) keys to each player detection.

        Args:
            tracks: Nested tracking dictionary structured as
                {object_type: [{track_id: {track_info}}]}.
                Each track must have a 'position_transformed' key.
        """
        total_distance: Dict[str, Dict[int, float]] = {}

        for object_type, object_tracks in tracks.items():
            if object_type in ("ball", "referees"):
                continue

            number_of_frames = len(object_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id in object_tracks[frame_num]:
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get("position_transformed")
                    end_position = object_tracks[last_frame][track_id].get("position_transformed")

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    if time_elapsed <= 0:
                        continue

                    speed_mps = distance_covered / time_elapsed
                    speed_kmh = speed_mps * 3.6

                    # Accumulate total distance
                    if object_type not in total_distance:
                        total_distance[object_type] = {}
                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0.0

                    total_distance[object_type][track_id] += distance_covered

                    # Apply to all frames in the window
                    for batch_frame in range(frame_num, last_frame):
                        if track_id not in tracks[object_type][batch_frame]:
                            continue
                        tracks[object_type][batch_frame][track_id]["speed"] = speed_kmh
                        tracks[object_type][batch_frame][track_id]["distance"] = total_distance[
                            object_type
                        ][track_id]

    def draw_speed_and_distance(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict[int, dict]]],
    ) -> List[np.ndarray]:
        """
        Draw speed and distance annotations on video frames.

        Renders the current speed (km/h) and accumulated distance (m)
        below each player's bounding box.

        Args:
            frames: List of BGR video frames.
            tracks: Tracking dictionary with 'speed' and 'distance' keys.

        Returns:
            List of annotated frames with speed/distance overlays.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object_type, object_tracks in tracks.items():
                if object_type in ("ball", "referees"):
                    continue

                if frame_num >= len(object_tracks):
                    continue

                for _, track_info in object_tracks[frame_num].items():
                    speed = track_info.get("speed")
                    distance = track_info.get("distance")

                    if speed is None or distance is None:
                        continue

                    bbox = track_info.get("bbox")
                    if bbox is None:
                        continue

                    position = get_foot_position(bbox)
                    text_x, text_y = int(position[0]), int(position[1]) + 40

                    cv2.putText(
                        frame,
                        f"{speed:.2f} km/h",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{distance:.2f} m",
                        (text_x, text_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )

            output_frames.append(frame)

        return output_frames
