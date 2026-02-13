"""
Player-ball assignment module for sports analytics.

Determines which player is in possession of the ball by
computing the minimum distance between each player's position
and the ball's position.

Author: Malav Patel
"""

import logging
from typing import Dict

from src.utils import get_center_of_bbox, measure_distance

logger = logging.getLogger(__name__)


class PlayerBallAssigner:
    """
    Assigns ball possession to the nearest player within a threshold distance.

    The assignment uses the center of each player's bounding box and
    the center of the ball's bounding box to compute Euclidean distances.

    Attributes:
        max_player_ball_distance: Maximum pixel distance for ball assignment.
    """

    def __init__(self, max_player_ball_distance: float = 70.0) -> None:
        """
        Initialize the player-ball assigner.

        Args:
            max_player_ball_distance: Maximum distance (in pixels) between
                a player and the ball to consider possession. Default is 70.
        """
        self.max_player_ball_distance = max_player_ball_distance

    def assign_ball_to_player(
        self,
        players: Dict[int, dict],
        ball_bbox: list[int],
    ) -> int:
        """
        Assign the ball to the nearest player within the threshold distance.

        Args:
            players: Dictionary mapping player IDs to their track data,
                each containing a 'bbox' key with [x1, y1, x2, y2].
            ball_bbox: Bounding box of the ball [x1, y1, x2, y2].

        Returns:
            Player ID of the assigned player, or -1 if no player is
            close enough to the ball.
        """
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = float("inf")
        assigned_player = -1

        for player_id, player_data in players.items():
            player_bbox = player_data.get("bbox")
            if player_bbox is None:
                continue

            # Use bottom-left and bottom-right corners of player bbox
            # to better estimate foot-level proximity to the ball
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
