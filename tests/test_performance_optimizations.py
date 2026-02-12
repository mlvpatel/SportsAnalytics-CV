"""Tests for performance optimizations."""

import numpy as np
import pytest


def test_numpy_interpolation_logic():
    """Test the numpy interpolation logic directly."""
    # Simulate the interpolation logic from interpolate_ball_positions
    
    # Test data with gaps
    ball_positions_data = [
        [100, 200, 120, 220],
        [np.nan, np.nan, np.nan, np.nan],  # Missing
        [np.nan, np.nan, np.nan, np.nan],  # Missing
        [140, 240, 160, 260],
        [150, 250, 170, 270],
    ]
    
    ball_positions_array = np.array(ball_positions_data, dtype=float)
    
    # Interpolate missing values using numpy
    for col in range(ball_positions_array.shape[1]):
        col_data = ball_positions_array[:, col]
        valid_indices = np.where(~np.isnan(col_data))[0]
        
        if len(valid_indices) > 1:
            all_indices = np.arange(len(col_data))
            ball_positions_array[:, col] = np.interp(
                all_indices,
                valid_indices,
                col_data[valid_indices]
            )
    
    # Verify interpolation worked
    # Second position should be interpolated between first and fourth
    assert not np.isnan(ball_positions_array[1, 0])
    assert not np.isnan(ball_positions_array[2, 0])
    
    # Values should be between bounds
    assert 100 <= ball_positions_array[1, 0] <= 140
    assert 100 <= ball_positions_array[2, 0] <= 140


def test_numpy_interpolation_all_nan():
    """Test interpolation when all values are NaN."""
    ball_positions_array = np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ], dtype=float)
    
    # Interpolate missing values
    for col in range(ball_positions_array.shape[1]):
        col_data = ball_positions_array[:, col]
        valid_indices = np.where(~np.isnan(col_data))[0]
        
        if len(valid_indices) > 1:
            all_indices = np.arange(len(col_data))
            ball_positions_array[:, col] = np.interp(
                all_indices,
                valid_indices,
                col_data[valid_indices]
            )
    
    # All should still be NaN since no valid data
    assert np.all(np.isnan(ball_positions_array))


def test_numpy_interpolation_no_gaps():
    """Test interpolation when there are no gaps."""
    original = np.array([
        [100, 200, 120, 220],
        [110, 210, 130, 230],
        [120, 220, 140, 240],
    ], dtype=float)
    
    ball_positions_array = original.copy()
    
    # Interpolate (should do nothing since no NaN values)
    for col in range(ball_positions_array.shape[1]):
        col_data = ball_positions_array[:, col]
        valid_indices = np.where(~np.isnan(col_data))[0]
        
        if len(valid_indices) > 1:
            all_indices = np.arange(len(col_data))
            ball_positions_array[:, col] = np.interp(
                all_indices,
                valid_indices,
                col_data[valid_indices]
            )
    
    # Should be unchanged
    np.testing.assert_array_almost_equal(ball_positions_array, original)


def test_video_properties_function_signature():
    """Test that video properties function has correct structure."""
    # Import the function to verify it exists
    try:
        from src.utils import get_video_properties
        
        # Verify function is callable
        assert callable(get_video_properties)
        
        # Check function signature
        import inspect
        sig = inspect.signature(get_video_properties)
        assert 'video_path' in sig.parameters
        
    except ImportError:
        pytest.skip("Module dependencies not available")


def test_speed_estimator_accepts_frame_rate():
    """Test that speed estimator can be initialized with custom frame rate."""
    try:
        from src.speed_distance import SpeedAndDistance_Estimator
        
        # Test with default frame rate
        estimator_default = SpeedAndDistance_Estimator()
        assert estimator_default.frame_rate == 24
        
        # Test with custom frame rate
        estimator_30fps = SpeedAndDistance_Estimator(frame_rate=30)
        assert estimator_30fps.frame_rate == 30
        
        estimator_60fps = SpeedAndDistance_Estimator(frame_rate=60)
        assert estimator_60fps.frame_rate == 60
        
    except ImportError:
        pytest.skip("Module dependencies not available")


def test_save_video_accepts_fps_parameter():
    """Test that save_video function accepts fps parameter."""
    try:
        from src.utils import save_video
        import inspect
        
        # Check function signature
        sig = inspect.signature(save_video)
        param_names = list(sig.parameters.keys())
        
        # Should have fps parameter
        assert 'fps' in param_names
        
        # Check default value
        assert sig.parameters['fps'].default == 24
        
    except ImportError:
        pytest.skip("Module dependencies not available")


def test_interpolation_edge_case_single_valid():
    """Test interpolation with only one valid value."""
    ball_positions_array = np.array([
        [np.nan, np.nan, np.nan, np.nan],
        [100, 200, 120, 220],  # Only one valid value
        [np.nan, np.nan, np.nan, np.nan],
    ], dtype=float)
    
    # Interpolate missing values
    for col in range(ball_positions_array.shape[1]):
        col_data = ball_positions_array[:, col]
        valid_indices = np.where(~np.isnan(col_data))[0]
        
        if len(valid_indices) > 1:
            all_indices = np.arange(len(col_data))
            ball_positions_array[:, col] = np.interp(
                all_indices,
                valid_indices,
                col_data[valid_indices]
            )
        elif len(valid_indices) == 1:
            # Fill all with the single valid value
            ball_positions_array[:, col] = col_data[valid_indices[0]]
    
    # All rows should now have the valid value
    np.testing.assert_array_equal(ball_positions_array[0], [100, 200, 120, 220])
    np.testing.assert_array_equal(ball_positions_array[1], [100, 200, 120, 220])
    np.testing.assert_array_equal(ball_positions_array[2], [100, 200, 120, 220])

