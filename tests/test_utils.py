"""Basic tests for utility functions."""

from src.utils.bbox_utils import get_center_of_bbox, get_bbox_width, measure_distance


def test_get_center_of_bbox():
    bbox = [100, 200, 300, 400]
    cx, cy = get_center_of_bbox(bbox)
    assert cx == 200
    assert cy == 300


def test_get_bbox_width():
    bbox = [100, 200, 300, 400]
    assert get_bbox_width(bbox) == 200


def test_measure_distance():
    p1 = (0, 0)
    p2 = (3, 4)
    assert measure_distance(p1, p2) == 5.0
