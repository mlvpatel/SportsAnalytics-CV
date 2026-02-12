import numpy as np
from src.utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player = -1

        # Vectorize distance calculation for better performance
        if players:
            player_ids = list(players.keys())
            player_bboxes = [players[pid]["bbox"] for pid in player_ids]
            
            # Calculate distances to both left and right edges for all players at once
            left_positions = np.array([(bbox[0], bbox[-1]) for bbox in player_bboxes])
            right_positions = np.array([(bbox[2], bbox[-1]) for bbox in player_bboxes])
            ball_pos = np.array(ball_position)
            
            # Calculate distances using vectorized NumPy operations
            distances_left = np.linalg.norm(left_positions - ball_pos, axis=1)
            distances_right = np.linalg.norm(right_positions - ball_pos, axis=1)
            distances = np.minimum(distances_left, distances_right)
            
            # Find the player with minimum distance within threshold
            valid_mask = distances < self.max_player_ball_distance
            if valid_mask.any():
                valid_distances = np.where(valid_mask, distances, np.inf)
                min_idx = np.argmin(valid_distances)
                if valid_distances[min_idx] < np.inf:
                    assigned_player = player_ids[min_idx]

        return assigned_player
