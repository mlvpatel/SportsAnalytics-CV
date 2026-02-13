"""
Unit tests for Tracker module.

Tests the non-GPU-dependent methods of Tracker.
These tests are skipped if supervision/ultralytics are not installed,
since they are heavy GPU dependencies not available in CI.

Author: Malav Patel
"""

import numpy as np
import pytest

try:
    import supervision  # noqa: F401
    import ultralytics  # noqa: F401

    HAS_GPU_DEPS = True
except ImportError:
    HAS_GPU_DEPS = False

skip_without_gpu_deps = pytest.mark.skipif(
    not HAS_GPU_DEPS,
    reason="Requires supervision and ultralytics (GPU dependencies)",
)


@skip_without_gpu_deps
class TestTrackerInit:
    """Tests for Tracker initialization."""

    def test_missing_model_file(self):
        from src.trackers.tracker import Tracker

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            Tracker("/nonexistent/model.pt")


@skip_without_gpu_deps
class TestTrackerMethods:
    """Tests for Tracker methods using a mocked model."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with a mocked YOLO model."""
        from unittest.mock import MagicMock, patch

        with patch("src.trackers.tracker.YOLO") as mock_yolo:
            with patch("src.trackers.tracker.os.path.exists", return_value=True):
                mock_yolo.return_value = MagicMock()
                from src.trackers.tracker import Tracker

                return Tracker("fake_model.pt")

    def test_add_position_to_tracks_player(self, tracker, sample_tracks):
        tracker.add_position_to_tracks(sample_tracks)
        player = sample_tracks["players"][0][1]
        assert "position" in player
        assert player["position"] == (150, 400)

    def test_add_position_to_tracks_ball(self, tracker, sample_tracks):
        tracker.add_position_to_tracks(sample_tracks)
        ball = sample_tracks["ball"][0][1]
        assert ball["position"] == (160, 310)

    def test_interpolate_ball_positions(self, tracker):
        ball_positions = [
            {1: {"bbox": [10, 10, 20, 20]}},
            {},
            {1: {"bbox": [30, 30, 40, 40]}},
        ]
        result = tracker.interpolate_ball_positions(ball_positions)
        assert len(result) == 3
        assert 1 in result[1]
        assert "bbox" in result[1][1]
        bbox = result[1][1]["bbox"]
        assert bbox[0] == pytest.approx(20.0)

    def test_interpolate_all_missing(self, tracker):
        ball_positions = [{}, {}, {}]
        result = tracker.interpolate_ball_positions(ball_positions)
        assert len(result) == 3

    def test_draw_ellipse(self, tracker, sample_frame, sample_bbox):
        result = tracker.draw_ellipse(sample_frame, sample_bbox, (0, 255, 0), 5)
        assert result.shape == sample_frame.shape

    def test_draw_ellipse_no_track_id(self, tracker, sample_frame, sample_bbox):
        result = tracker.draw_ellipse(sample_frame, sample_bbox, (0, 255, 0))
        assert result.shape == sample_frame.shape

    def test_draw_triangle(self, tracker, sample_frame, sample_bbox):
        result = tracker.draw_triangle(sample_frame, sample_bbox, (0, 0, 255))
        assert result.shape == sample_frame.shape

    def test_draw_team_ball_control(self, tracker, sample_frame):
        team_ball_control = np.array([1, 1, 2, 1, 2])
        result = tracker.draw_team_ball_control(sample_frame, 4, team_ball_control)
        assert result.shape == sample_frame.shape

    def test_draw_team_ball_control_empty(self, tracker, sample_frame):
        team_ball_control = np.array([0, 0, 0])
        result = tracker.draw_team_ball_control(sample_frame, 2, team_ball_control)
        assert result.shape == sample_frame.shape

    def test_draw_annotations(self, tracker, sample_frame, sample_tracks):
        team_ball_control = np.array([1])
        output = tracker.draw_annotations([sample_frame], sample_tracks, team_ball_control)
        assert len(output) == 1
        assert output[0].shape == sample_frame.shape
