"""
Unit tests for CameraMovementEstimator module.

Author: Malav Patel
"""

import numpy as np
import pytest


class TestCameraMovementEstimator:
    """Tests for CameraMovementEstimator class."""

    def test_initialization(self, sample_frame):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        estimator = CameraMovementEstimator(sample_frame)
        assert estimator.minimum_distance == 5.0
        assert estimator.lk_params is not None
        assert estimator.features is not None

    def test_initialization_custom_distance(self, sample_frame):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        estimator = CameraMovementEstimator(sample_frame, minimum_distance=10.0)
        assert estimator.minimum_distance == 10.0

    def test_initialization_empty_frame(self):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        with pytest.raises(ValueError, match="empty"):
            CameraMovementEstimator(np.array([]))

    def test_initialization_none_frame(self):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        with pytest.raises(ValueError, match="empty"):
            CameraMovementEstimator(None)

    def test_get_camera_movement_returns_list(self, sample_frame):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        estimator = CameraMovementEstimator(sample_frame)
        frames = [sample_frame, sample_frame.copy()]
        movement = estimator.get_camera_movement(frames)

        assert len(movement) == 2
        assert movement[0] == [0, 0]  # First frame always [0, 0]

    def test_add_adjust_positions_to_tracks(self, sample_frame, sample_tracks):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        estimator = CameraMovementEstimator(sample_frame)
        camera_movement = [[10, 5]]

        estimator.add_adjust_positions_to_tracks(sample_tracks, camera_movement)

        player = sample_tracks["players"][0][1]
        assert "position_adjusted" in player
        assert player["position_adjusted"] == (140, 395)  # 150-10, 400-5

    def test_add_adjust_positions_with_no_position(self, sample_frame):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        estimator = CameraMovementEstimator(sample_frame)
        tracks = {"players": [{1: {"bbox": [0, 0, 10, 10]}}]}
        camera_movement = [[0, 0]]

        estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        assert tracks["players"][0][1]["position_adjusted"] is None

    def test_draw_camera_movement(self, sample_frame):
        from src.camera_movement.camera_movement_estimator import CameraMovementEstimator

        estimator = CameraMovementEstimator(sample_frame)
        frames = [sample_frame]
        movement = [[5.5, 3.2]]

        output = estimator.draw_camera_movement(frames, movement)
        assert len(output) == 1
        assert output[0].shape == sample_frame.shape
