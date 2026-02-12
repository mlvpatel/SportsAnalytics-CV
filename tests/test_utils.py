"""Basic tests for utility functions."""

import math


def test_get_center_of_bbox():
    """Test bounding box center calculation."""
    # Import inline to avoid importing the full src package
    from src.utils.bbox_utils import get_center_of_bbox

    bbox = [100, 200, 300, 400]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 200
    assert cy == 300


def test_get_bbox_width():
    """Test bounding box width calculation."""
    from src.utils.bbox_utils import get_bbox_width

    bbox = [100, 200, 300, 400]
    assert get_bbox_width(bbox) == 200


def test_measure_distance():
    """Test Euclidean distance measurement."""
    from src.utils.bbox_utils import measure_distance

    p1 = (0, 0)
    p2 = (3, 4)
    assert measure_distance(p1, p2) == 5.0


def test_get_foot_position():
    """Test foot position calculation from bounding box."""
    from src.utils.bbox_utils import get_foot_position

    bbox = [100, 200, 300, 400]
    x, y = get_foot_position(bbox)
    assert x == 200  # center x
    assert y == 400  # bottom y


def test_measure_distance_same_point():
    """Test distance between same point is 0."""
    from src.utils.bbox_utils import measure_distance

    p1 = (5, 5)
    assert measure_distance(p1, p1) == 0.0
