"""
View transformation module for sports analytics.

Transforms 2D pixel coordinates from the camera perspective
to a top-down court view using perspective transformation.
This enables accurate real-world distance and speed calculations.

Author: Malav Patel
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ViewTransformer:
    """
    Transforms pixel coordinates to real-world court coordinates.

    Uses OpenCV perspective transformation to map camera-view positions
    to a normalized top-down court view. The court dimensions are based
    on a standard football pitch segment.

    Attributes:
        court_width: Width of the court in meters.
        court_length: Length of the visible court segment in meters.
        pixel_vertices: Four corner points in the camera view.
        target_vertices: Corresponding four corner points in the court view.
    """

    def __init__(
        self,
        pixel_vertices: Optional[np.ndarray] = None,
        court_width: float = 68.0,
        court_length: float = 23.32,
    ) -> None:
        """
        Initialize the view transformer.

        Args:
            pixel_vertices: Four corner points in pixel coordinates as a
                (4, 2) numpy array. If None, uses default values calibrated
                for the sample match video.
            court_width: Width of the court in meters. Default is 68m
                (standard football pitch width).
            court_length: Length of the visible court segment in meters.
        """
        self.court_width = court_width
        self.court_length = court_length

        if pixel_vertices is None:
            self.pixel_vertices = np.array(
                [[110, 1035], [265, 275], [910, 260], [1640, 915]],
                dtype=np.float32,
            )
        else:
            self.pixel_vertices = pixel_vertices.astype(np.float32)

        self.target_vertices = np.array(
            [
                [0, court_width],
                [0, 0],
                [court_length, 0],
                [court_length, court_width],
            ],
            dtype=np.float32,
        )

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

        logger.debug(f"ViewTransformer initialized: court {court_length}m x {court_width}m")

    def transform_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a single point from pixel to court coordinates.

        Only transforms points that fall within the defined pixel polygon.
        Points outside the court boundaries return None.

        Args:
            point: A 2D point as a numpy array [x, y].

        Returns:
            Transformed point as a (1, 2) numpy array, or None if the
            point is outside the court boundaries.
        """
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0

        if not is_inside:
            return None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks: Dict[str, List[Dict[int, dict]]]) -> None:
        """
        Add transformed court positions to all tracked objects.

        Iterates through all tracks and adds a 'position_transformed'
        key with the court-view coordinates for each detection.

        Args:
            tracks: Nested tracking dictionary structured as
                {object_type: [{track_id: {track_info}}]}.
                Each track_info must contain a 'position_adjusted' key.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get("position_adjusted")
                    if position is None:
                        tracks[object_type][frame_num][track_id]["position_transformed"] = None
                        continue

                    position = np.array(position)
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()

                    tracks[object_type][frame_num][track_id][
                        "position_transformed"
                    ] = position_transformed
