"""
Unit tests for bbox_utils module.

Author: Malav Patel
"""

import pytest


class TestGetCenterOfBbox:
    """Tests for get_center_of_bbox function."""

    def test_standard_bbox(self):
        from src.utils.bbox_utils import get_center_of_bbox

        assert get_center_of_bbox([0, 0, 100, 100]) == (50, 50)

    def test_asymmetric_bbox(self):
        from src.utils.bbox_utils import get_center_of_bbox

        assert get_center_of_bbox([100, 200, 300, 400]) == (200, 300)

    def test_invalid_bbox_length(self):
        from src.utils.bbox_utils import get_center_of_bbox

        with pytest.raises(ValueError, match="Expected 4"):
            get_center_of_bbox([1, 2, 3])


class TestGetBboxWidth:
    """Tests for get_bbox_width function."""

    def test_standard_width(self):
        from src.utils.bbox_utils import get_bbox_width

        assert get_bbox_width([100, 200, 300, 400]) == 200

    def test_zero_width(self):
        from src.utils.bbox_utils import get_bbox_width

        assert get_bbox_width([100, 200, 100, 400]) == 0


class TestMeasureDistance:
    """Tests for measure_distance function."""

    def test_same_point(self):
        from src.utils.bbox_utils import measure_distance

        assert measure_distance((5, 5), (5, 5)) == 0.0

    def test_horizontal_distance(self):
        from src.utils.bbox_utils import measure_distance

        assert measure_distance((0, 0), (3, 0)) == 3.0

    def test_vertical_distance(self):
        from src.utils.bbox_utils import measure_distance

        assert measure_distance((0, 0), (0, 4)) == 4.0

    def test_diagonal_distance(self):
        from src.utils.bbox_utils import measure_distance

        assert measure_distance((0, 0), (3, 4)) == 5.0


class TestMeasureXYDistance:
    """Tests for measure_xy_distance function."""

    def test_positive_displacement(self):
        from src.utils.bbox_utils import measure_xy_distance

        dx, dy = measure_xy_distance((10, 20), (5, 10))
        assert dx == 5 and dy == 10

    def test_zero_displacement(self):
        from src.utils.bbox_utils import measure_xy_distance

        dx, dy = measure_xy_distance((5, 5), (5, 5))
        assert dx == 0 and dy == 0


class TestGetFootPosition:
    """Tests for get_foot_position function."""

    def test_standard_bbox(self):
        from src.utils.bbox_utils import get_foot_position

        x, y = get_foot_position([100, 200, 300, 400])
        assert x == 200  # center x
        assert y == 400  # bottom y

    def test_small_bbox(self):
        from src.utils.bbox_utils import get_foot_position

        x, y = get_foot_position([0, 0, 10, 10])
        assert x == 5 and y == 10
