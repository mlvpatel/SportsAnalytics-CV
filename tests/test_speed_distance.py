"""
Unit tests for SpeedDistanceEstimator module.

Author: Malav Patel
"""

import numpy as np
import pytest


class TestSpeedDistanceEstimator:
    """Tests for SpeedDistanceEstimator class."""

    def test_initialization(self):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        estimator = SpeedDistanceEstimator()
        assert estimator.frame_window == 5
        assert estimator.frame_rate == 24

    def test_custom_initialization(self):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        estimator = SpeedDistanceEstimator(frame_window=10, frame_rate=30)
        assert estimator.frame_window == 10
        assert estimator.frame_rate == 30

    def test_invalid_frame_rate(self):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        with pytest.raises(ValueError, match="frame_rate"):
            SpeedDistanceEstimator(frame_rate=0)

    def test_invalid_frame_window(self):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        with pytest.raises(ValueError, match="frame_window"):
            SpeedDistanceEstimator(frame_window=-1)

    def test_add_speed_and_distance(self, multi_frame_tracks):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        estimator = SpeedDistanceEstimator(frame_window=5, frame_rate=24)
        estimator.add_speed_and_distance_to_tracks(multi_frame_tracks)

        # Check that speed was calculated for player 1
        player = multi_frame_tracks["players"][0].get(1, {})
        assert "speed" in player
        assert "distance" in player
        assert player["speed"] > 0

    def test_skips_ball_and_referees(self, multi_frame_tracks):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        estimator = SpeedDistanceEstimator(frame_window=5, frame_rate=24)
        estimator.add_speed_and_distance_to_tracks(multi_frame_tracks)

        # Ball and referees tracks should remain unchanged
        for frame in multi_frame_tracks["ball"]:
            for _, info in frame.items():
                assert "speed" not in info

    def test_draw_speed_and_distance(self, multi_frame_tracks):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        estimator = SpeedDistanceEstimator(frame_window=5, frame_rate=24)
        estimator.add_speed_and_distance_to_tracks(multi_frame_tracks)

        frames = [np.zeros((500, 500, 3), dtype=np.uint8) for _ in range(6)]
        output = estimator.draw_speed_and_distance(frames, multi_frame_tracks)
        assert len(output) == 6

    def test_handles_none_positions(self):
        from src.speed_distance.speed_and_distance_estimator import SpeedDistanceEstimator

        estimator = SpeedDistanceEstimator(frame_window=2, frame_rate=24)
        tracks = {
            "players": [
                {1: {"bbox": [0, 0, 10, 10], "position_transformed": None}},
                {1: {"bbox": [0, 0, 10, 10], "position_transformed": None}},
                {1: {"bbox": [0, 0, 10, 10], "position_transformed": [5, 0]}},
            ],
            "ball": [{}, {}, {}],
            "referees": [{}, {}, {}],
        }
        # Should not raise
        estimator.add_speed_and_distance_to_tracks(tracks)
