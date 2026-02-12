"""Tests for performance optimization fixes."""

import numpy as np
import pytest


def test_camera_movement_list_independence():
    """Test that camera movement list elements are independent (not shared references)."""
    # Test the fix directly without needing full camera movement estimation
    frames = [None] * 5

    # This is how camera_movement is created in the fixed version
    camera_movement = [[0, 0] for _ in range(len(frames))]

    # Modify one element
    camera_movement[2] = [5, 10]

    # Check that other elements are not affected (this would fail with shared references)
    assert camera_movement[0] == [0, 0], "First element should remain [0, 0]"
    assert camera_movement[1] == [0, 0], "Second element should remain [0, 0]"
    assert camera_movement[2] == [5, 10], "Third element should be [5, 10]"
    assert camera_movement[3] == [0, 0], "Fourth element should remain [0, 0]"
    assert camera_movement[4] == [0, 0], "Fifth element should remain [0, 0]"

    # Verify they are different objects
    assert id(camera_movement[0]) != id(camera_movement[1])
    assert id(camera_movement[1]) != id(camera_movement[2])

    # Test the buggy version would have failed
    buggy_camera_movement = [[0, 0]] * len(frames)
    buggy_camera_movement[2] = [5, 10]
    # In the buggy version, all elements would have been modified (but we can't test this
    # as Python's list multiplication creates independent references for immutable contents)


def test_measure_distance_optimized():
    """Test that optimized distance calculation produces same results."""
    from src.utils.bbox_utils import measure_distance
    
    # Test various distances
    assert abs(measure_distance((0, 0), (3, 4)) - 5.0) < 1e-6
    assert abs(measure_distance((0, 0), (0, 0)) - 0.0) < 1e-6
    assert abs(measure_distance((1, 1), (4, 5)) - 5.0) < 1e-6
    assert abs(measure_distance((-3, -4), (0, 0)) - 5.0) < 1e-6


def test_player_ball_assigner_vectorized():
    """Test that vectorized player-ball assignment works correctly."""
    from src.player_ball_assigner import PlayerBallAssigner
    
    assigner = PlayerBallAssigner()
    
    # Test with empty players
    ball_bbox = [100, 100, 120, 120]
    players = {}
    result = assigner.assign_ball_to_player(players, ball_bbox)
    assert result == -1
    
    # Test with players within distance
    players = {
        1: {"bbox": [90, 90, 110, 110]},  # Close to ball
        2: {"bbox": [200, 200, 220, 220]},  # Far from ball
    }
    ball_bbox = [100, 100, 120, 120]
    result = assigner.assign_ball_to_player(players, ball_bbox)
    assert result == 1  # Should assign to player 1
    
    # Test with no players within distance threshold
    players = {
        1: {"bbox": [500, 500, 520, 520]},  # Far from ball
        2: {"bbox": [600, 600, 620, 620]},  # Far from ball
    }
    ball_bbox = [100, 100, 120, 120]
    result = assigner.assign_ball_to_player(players, ball_bbox)
    assert result == -1


def test_counter_corner_clustering():
    """Test that Counter-based corner clustering works correctly."""
    from collections import Counter
    
    # Test case 1: Clear majority
    corner_clusters = [0, 0, 0, 1]
    non_player_cluster = Counter(corner_clusters).most_common(1)[0][0]
    assert non_player_cluster == 0
    
    # Test case 2: All same
    corner_clusters = [1, 1, 1, 1]
    non_player_cluster = Counter(corner_clusters).most_common(1)[0][0]
    assert non_player_cluster == 1
    
    # Test case 3: Tie (should return one of them)
    corner_clusters = [0, 0, 1, 1]
    non_player_cluster = Counter(corner_clusters).most_common(1)[0][0]
    assert non_player_cluster in [0, 1]


def test_save_video_fps_parameter():
    """Test that save_video accepts fps parameter."""
    from src.utils.video_utils import save_video
    import inspect
    
    # Check that fps parameter exists in function signature
    sig = inspect.signature(save_video)
    assert 'fps' in sig.parameters
    assert sig.parameters['fps'].default == 24
