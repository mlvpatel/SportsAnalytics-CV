"""
Unit tests for PlayerBallAssigner module.

Author: Malav Patel
"""


class TestPlayerBallAssigner:
    """Tests for PlayerBallAssigner class."""

    def test_initialization_default(self):
        from src.player_ball_assigner.player_ball_assigner import PlayerBallAssigner

        assigner = PlayerBallAssigner()
        assert assigner.max_player_ball_distance == 70.0

    def test_initialization_custom(self):
        from src.player_ball_assigner.player_ball_assigner import PlayerBallAssigner

        assigner = PlayerBallAssigner(max_player_ball_distance=100.0)
        assert assigner.max_player_ball_distance == 100.0

    def test_assign_ball_to_nearest_player(self):
        from src.player_ball_assigner.player_ball_assigner import PlayerBallAssigner

        assigner = PlayerBallAssigner(max_player_ball_distance=100)
        players = {
            1: {"bbox": [100, 200, 150, 250]},  # Far from ball
            2: {"bbox": [290, 290, 310, 310]},  # Close to ball
        }
        ball_bbox = [295, 295, 305, 305]

        result = assigner.assign_ball_to_player(players, ball_bbox)
        assert result == 2

    def test_no_player_close_enough(self):
        from src.player_ball_assigner.player_ball_assigner import PlayerBallAssigner

        assigner = PlayerBallAssigner(max_player_ball_distance=10)
        players = {
            1: {"bbox": [0, 0, 50, 50]},
            2: {"bbox": [500, 500, 550, 550]},
        }
        ball_bbox = [250, 250, 260, 260]

        result = assigner.assign_ball_to_player(players, ball_bbox)
        assert result == -1

    def test_empty_players(self):
        from src.player_ball_assigner.player_ball_assigner import PlayerBallAssigner

        assigner = PlayerBallAssigner()
        result = assigner.assign_ball_to_player({}, [100, 100, 110, 110])
        assert result == -1

    def test_player_without_bbox(self):
        from src.player_ball_assigner.player_ball_assigner import PlayerBallAssigner

        assigner = PlayerBallAssigner()
        players = {1: {"team": 1}}  # No bbox key
        result = assigner.assign_ball_to_player(players, [100, 100, 110, 110])
        assert result == -1
