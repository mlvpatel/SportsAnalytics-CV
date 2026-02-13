"""
Shared pytest fixtures for SportsAnalytics-CV tests.

Author: Malav Patel
"""

import numpy as np
import pytest


@pytest.fixture
def sample_frame():
    """Generate a sample 1920x1080 BGR frame."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def small_frame():
    """Generate a small 100x100 BGR frame for fast tests."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Standard bounding box [x1, y1, x2, y2]."""
    return [100, 200, 200, 400]


@pytest.fixture
def sample_tracks():
    """Minimal tracking dictionary for testing."""
    return {
        "players": [
            {
                1: {"bbox": [100, 200, 200, 400], "position": (150, 400)},
                2: {"bbox": [300, 200, 400, 400], "position": (350, 400)},
            }
        ],
        "referees": [{3: {"bbox": [500, 200, 600, 400], "position": (550, 400)}}],
        "ball": [{1: {"bbox": [150, 300, 170, 320], "position": (160, 310)}}],
    }


@pytest.fixture
def multi_frame_tracks():
    """Multi-frame tracking dictionary for speed/distance tests."""
    return {
        "players": [
            {
                1: {
                    "bbox": [100, 200, 200, 400],
                    "position_transformed": [0, 0],
                },
            },
            {
                1: {
                    "bbox": [110, 200, 210, 400],
                    "position_transformed": [5, 0],
                },
            },
            {
                1: {
                    "bbox": [120, 200, 220, 400],
                    "position_transformed": [10, 0],
                },
            },
            {
                1: {
                    "bbox": [130, 200, 230, 400],
                    "position_transformed": [15, 0],
                },
            },
            {
                1: {
                    "bbox": [140, 200, 240, 400],
                    "position_transformed": [20, 0],
                },
            },
            {
                1: {
                    "bbox": [150, 200, 250, 400],
                    "position_transformed": [25, 0],
                },
            },
        ],
        "ball": [{}, {}, {}, {}, {}, {}],
        "referees": [{}, {}, {}, {}, {}, {}],
    }
