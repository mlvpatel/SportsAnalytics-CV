"""
Camera movement estimation module for sports analytics.

Uses Lucas-Kanade optical flow to estimate camera panning
between consecutive frames. This allows compensating player
positions for camera movement, enabling accurate speed and
distance calculations.

Author: Malav Patel
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.utils import measure_distance, measure_xy_distance

logger = logging.getLogger(__name__)


class CameraMovementEstimator:
    """
    Estimates camera movement using sparse optical flow.

    Detects good features to track in designated screen regions
    (left and right edges) and tracks their displacement between
    frames using Lucas-Kanade optical flow.

    Attributes:
        minimum_distance: Minimum feature displacement (pixels) to
            register as camera movement.
        lk_params: Parameters for Lucas-Kanade optical flow.
        features: Parameters for goodFeaturesToTrack detection.
    """

    def __init__(
        self,
        frame: np.ndarray,
        minimum_distance: float = 5.0,
    ) -> None:
        """
        Initialize the camera movement estimator with the first frame.

        Sets up the feature detection mask and optical flow parameters.
        Features are detected only in the left and right edge regions
        of the frame to capture stadium/field boundaries.

        Args:
            frame: First video frame as a BGR numpy array.
            minimum_distance: Minimum pixel displacement to count as
                camera movement. Default is 5.0.

        Raises:
            ValueError: If the frame is empty or has invalid dimensions.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Cannot initialize with an empty frame")

        self.minimum_distance = minimum_distance

        self.lk_params: Dict = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features: Dict = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

        logger.debug("CameraMovementEstimator initialized")

    def add_adjust_positions_to_tracks(
        self,
        tracks: Dict[str, List[Dict[int, dict]]],
        camera_movement_per_frame: List[List[float]],
    ) -> None:
        """
        Adjust all tracked positions by subtracting camera movement.

        Modifies tracks in-place, adding a 'position_adjusted' key
        to each detection with the camera-compensated position.

        Args:
            tracks: Nested tracking dictionary.
            camera_movement_per_frame: List of [dx, dy] camera
                movement values for each frame.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get("position")
                    if position is None:
                        tracks[object_type][frame_num][track_id]["position_adjusted"] = None
                        continue

                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object_type][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def get_camera_movement(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Calculate camera movement for each frame using optical flow.

        Tracks feature points between consecutive frames and measures
        their displacement. The maximum displacement determines the
        camera movement for that frame.

        Args:
            frames: List of BGR video frames.
            read_from_stub: If True, try to load cached results.
            stub_path: Path to the pickle cache file.

        Returns:
            List of [camera_movement_x, camera_movement_y] for each frame.

        Raises:
            IOError: If stub file exists but cannot be read.
        """
        # Try to load cached results
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    logger.info(f"Loaded camera movement from cache: {stub_path}")
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                logger.warning(f"Failed to load cache {stub_path}: {e}")

        camera_movement: List[List[float]] = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                old_gray = frame_gray.copy()
                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None:
                old_gray = frame_gray.copy()
                continue

            max_distance = 0.0
            camera_movement_x, camera_movement_y = 0.0, 0.0

            for new, old in zip(new_features, old_features):
                new_point = new.ravel()
                old_point = old.ravel()

                distance = measure_distance(new_point, old_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [
                    camera_movement_x,
                    camera_movement_y,
                ]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        # Save to cache
        if stub_path is not None:
            try:
                with open(stub_path, "wb") as f:
                    pickle.dump(camera_movement, f)
                logger.info(f"Saved camera movement cache: {stub_path}")
            except IOError as e:
                logger.warning(f"Failed to save cache {stub_path}: {e}")

        return camera_movement

    def draw_camera_movement(
        self,
        frames: List[np.ndarray],
        camera_movement_per_frame: List[List[float]],
    ) -> List[np.ndarray]:
        """
        Draw camera movement overlay on video frames.

        Renders a semi-transparent white box in the top-left corner
        showing the X and Y camera displacement values.

        Args:
            frames: List of BGR video frames.
            camera_movement_per_frame: List of [dx, dy] values per frame.

        Returns:
            List of annotated frames with camera movement overlay.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            output_frames.append(frame)

        return output_frames
