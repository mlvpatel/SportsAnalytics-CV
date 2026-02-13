"""
Team assignment module for sports analytics.

Uses K-Means clustering on jersey colors to identify two teams
and assign each detected player to their respective team.

Author: Malav Patel
"""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class TeamAssigner:
    """
    Assigns players to teams based on jersey color clustering.

    Uses a two-stage K-Means approach:
    1. Extract the dominant jersey color for each player (via top-half crop).
    2. Cluster all player colors into two groups to identify the two teams.

    Attributes:
        team_colors: Dictionary mapping team IDs (1, 2) to their RGB colors.
        player_team_dict: Cache of player ID → team ID assignments.
    """

    def __init__(self) -> None:
        """Initialize the team assigner with empty color/team mappings."""
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans: Optional[KMeans] = None

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Fit a 2-cluster K-Means model on an image's pixel colors.

        Args:
            image: BGR image as a numpy array (H, W, 3).

        Returns:
            Fitted KMeans model with 2 clusters.

        Raises:
            ValueError: If the image is empty or has invalid shape.
        """
        if image.size == 0:
            raise ValueError("Cannot cluster an empty image")

        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        """
        Extract the dominant jersey color for a player.

        Crops the player from the frame using the bounding box, takes
        the top half (to focus on the jersey, not shorts/legs), and
        clusters the pixel colors. The corner pixels are assumed to be
        background, so the non-corner cluster is the player color.

        Args:
            frame: Full video frame as a BGR numpy array.
            bbox: Player bounding box [x1, y1, x2, y2].

        Returns:
            RGB color of the player's jersey as a (3,) numpy array.

        Raises:
            ValueError: If the bounding box produces an empty crop.
        """
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        if image.size == 0:
            raise ValueError(
                f"Empty player crop from bbox {bbox}. " "Player may be partially out of frame."
            )

        top_half_image = image[0 : int(image.shape[0] / 2), :]

        if top_half_image.size == 0:
            raise ValueError("Top-half crop is empty, bbox may be too small")

        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_

        # Reshape labels to image dimensions
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identify background cluster from corner pixels
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame: np.ndarray, player_detections: Dict[int, dict]) -> None:
        """
        Determine team colors from the first frame's player detections.

        Extracts jersey colors for all detected players and clusters
        them into two team groups.

        Args:
            frame: First video frame as a BGR numpy array.
            player_detections: Dictionary mapping player IDs to their
                track data, each containing a 'bbox' key.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            try:
                player_color = self.get_player_color(frame, bbox)
                player_colors.append(player_color)
            except ValueError as e:
                logger.warning(f"Skipping player for team color: {e}")
                continue

        if len(player_colors) < 2:
            logger.error("Not enough valid player colors for team assignment")
            return

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(player_colors)

        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

        logger.info(
            f"Team colors assigned: Team 1={self.team_colors[1].astype(int)}, "
            f"Team 2={self.team_colors[2].astype(int)}"
        )

    def get_player_team(
        self,
        frame: np.ndarray,
        player_bbox: list[int],
        player_id: int,
    ) -> int:
        """
        Get the team assignment for a specific player.

        Uses cached assignments when available. For new players,
        extracts their jersey color and predicts the team using
        the fitted K-Means model.

        Args:
            frame: Current video frame as a BGR numpy array.
            player_bbox: Player bounding box [x1, y1, x2, y2].
            player_id: Unique identifier for the player.

        Returns:
            Team ID (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            logger.warning("Team colors not assigned yet, defaulting to team 1")
            return 1

        try:
            player_color = self.get_player_color(frame, player_bbox)
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1  # Convert from 0-indexed to 1-indexed
        except ValueError as e:
            logger.warning(f"Cannot determine team for player {player_id}: {e}")
            team_id = 1  # Default to team 1

        self.player_team_dict[player_id] = team_id
        return team_id
