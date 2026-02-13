"""
Bounding box utility functions for sports analytics.

Provides geometric calculations for bounding boxes including
center computation, width measurement, distance calculations,
and foot position estimation.

Author: Malav Patel
"""

from typing import Tuple


def get_center_of_bbox(bbox: list[int]) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Tuple of (center_x, center_y) as integers.

    Raises:
        ValueError: If bbox does not contain exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected 4 bbox coordinates, got {len(bbox)}")

    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list[int]) -> int:
    """
    Calculate the width of a bounding box.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Width of the bounding box in pixels.
    """
    return bbox[2] - bbox[0]


def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        Euclidean distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the x and y distance components between two points.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        Tuple of (dx, dy) distance components.
    """
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox: list[int]) -> Tuple[int, int]:
    """
    Estimate the foot position from a player's bounding box.

    The foot position is estimated as the bottom-center of the
    bounding box, which approximates where a player's feet
    contact the ground.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Tuple of (foot_x, foot_y) as integers.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
