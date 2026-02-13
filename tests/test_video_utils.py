"""
Unit tests for video_utils module.

Tests read_video and save_video with temporary files.

Author: Malav Patel
"""

import os

import cv2
import numpy as np
import pytest

from src.utils.video_utils import read_video, save_video


class TestReadVideo:
    """Tests for read_video function."""

    def test_read_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="Cannot open video"):
            read_video("/nonexistent/path/video.mp4")

    def test_read_invalid_file(self, tmp_path):
        # Create a text file pretending to be a video
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_text("not a video")

        with pytest.raises((FileNotFoundError, IOError)):
            read_video(str(fake_video))

    def test_read_valid_video(self, tmp_path):
        # Create a small valid video with OpenCV
        video_path = str(tmp_path / "test.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(video_path, fourcc, 24.0, (100, 100))

        for _ in range(5):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        frames = read_video(video_path)
        assert len(frames) == 5
        assert frames[0].shape == (100, 100, 3)


class TestSaveVideo:
    """Tests for save_video function."""

    def test_save_empty_frames(self, tmp_path):
        output_path = str(tmp_path / "empty.avi")
        with pytest.raises(ValueError, match="frame list is empty"):
            save_video([], output_path)

    def test_save_valid_frames(self, tmp_path):
        output_path = str(tmp_path / "output.avi")
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]

        save_video(frames, output_path)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_save_and_read_roundtrip(self, tmp_path):
        output_path = str(tmp_path / "roundtrip.avi")
        original_frames = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)
        ]

        save_video(original_frames, output_path)
        loaded_frames = read_video(output_path)

        assert len(loaded_frames) == len(original_frames)
        assert loaded_frames[0].shape == original_frames[0].shape
