"""
Unit tests for TeamAssigner module.

Author: Malav Patel
"""

import numpy as np
import pytest


class TestTeamAssigner:
    """Tests for TeamAssigner class."""

    def test_initialization(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        assert assigner.team_colors == {}
        assert assigner.player_team_dict == {}
        assert assigner.kmeans is None

    def test_get_clustering_model(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        # Create a simple 10x10 image with 2 distinct colors
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:5, :] = [255, 0, 0]  # Top half blue
        image[5:, :] = [0, 255, 0]  # Bottom half green

        kmeans = assigner.get_clustering_model(image)
        assert kmeans.n_clusters == 2
        assert len(kmeans.cluster_centers_) == 2

    def test_get_clustering_model_empty_image(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        empty_image = np.array([]).reshape(0, 0, 3).astype(np.uint8)

        with pytest.raises(ValueError, match="empty"):
            assigner.get_clustering_model(empty_image)

    def test_get_player_color(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        # Create a frame with a "player" region
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[100:300, 100:200] = [0, 0, 255]  # Red player region

        color = assigner.get_player_color(frame, [100, 100, 200, 300])
        assert color.shape == (3,)

    def test_get_player_color_empty_crop(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Empty"):
            assigner.get_player_color(frame, [50, 50, 50, 50])

    def test_get_player_team_default_without_kmeans(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        team = assigner.get_player_team(frame, [10, 10, 50, 50], 1)
        assert team == 1  # Default when kmeans not fitted

    def test_get_player_team_cached(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()
        assigner.player_team_dict[42] = 2

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        team = assigner.get_player_team(frame, [10, 10, 50, 50], 42)
        assert team == 2

    def test_assign_team_color_with_players(self):
        from src.team_assigner.team_assigner import TeamAssigner

        assigner = TeamAssigner()

        # Create frame with two distinctly colored "players"
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        frame[50:200, 50:150] = [255, 0, 0]  # Blue player region
        frame[50:200, 250:350] = [0, 0, 255]  # Red player region

        detections = {
            1: {"bbox": [50, 50, 150, 200]},
            2: {"bbox": [250, 50, 350, 200]},
        }

        assigner.assign_team_color(frame, detections)
        assert 1 in assigner.team_colors
        assert 2 in assigner.team_colors
        assert assigner.kmeans is not None
