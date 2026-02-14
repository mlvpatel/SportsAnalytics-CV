"""
Object tracking module for sports analytics.

Provides YOLO-based object detection with ByteTrack tracking
for players, referees, and the ball. Includes annotation
drawing functions for visualization.

Author: Malav Patel
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from src.utils import get_bbox_width, get_center_of_bbox, get_foot_position

logger = logging.getLogger(__name__)


class Tracker:
    """
    YOLO + ByteTrack object tracker for sports video analysis.

    Detects players, referees, and the ball using a fine-tuned YOLO
    model, then applies ByteTrack for consistent ID assignment across
    frames. Also includes methods for ball position interpolation and
    visual annotation rendering.

    Attributes:
        model: YOLO detection model.
        tracker: ByteTrack tracker instance.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize the tracker with a YOLO model.

        Args:
            model_path: Path to the YOLO model weights file (.pt).

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the model fails to load.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run 'python scripts/download_models.py' to download models."
            )

        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}") from e

        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: Dict[str, List[Dict[int, dict]]]) -> None:
        """
        Add position coordinates to all tracked objects.

        For the ball, uses the bounding box center. For players and
        referees, uses the foot position (bottom-center of bbox).

        Args:
            tracks: Nested tracking dictionary to modify in-place.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info.get("bbox")
                    if bbox is None:
                        continue

                    if object_type == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    tracks[object_type][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(
        self, ball_positions: List[Dict[int, dict]]
    ) -> List[Dict[int, dict]]:
        """
        Interpolate missing ball positions across frames.

        Uses pandas interpolation to fill gaps where the ball
        was not detected, then backfills any remaining NaN values
        at the start.

        Args:
            ball_positions: List of per-frame ball track dictionaries.

        Returns:
            Interpolated ball positions with continuous tracking.
        """
        ball_bboxes = []
        for x in ball_positions:
            bbox = x.get(1, {}).get("bbox")
            if bbox:
                ball_bboxes.append(bbox)
            else:
                ball_bboxes.append([np.nan, np.nan, np.nan, np.nan])

        df_ball = pd.DataFrame(ball_bboxes, columns=["x1", "y1", "x2", "y2"])

        # Interpolate and backfill missing values
        df_ball = df_ball.interpolate()
        df_ball = df_ball.bfill()

        interpolated = [{1: {"bbox": x}} for x in df_ball.to_numpy().tolist()]

        logger.debug(
            f"Interpolated ball positions: " f"{sum(1 for b in ball_bboxes if not b)} gaps filled"
        )
        return interpolated

    def detect_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 20,
        conf: float = 0.1,
    ) -> list:
        """
        Run YOLO detection on all frames in batches.

        Args:
            frames: List of BGR video frames.
            batch_size: Number of frames per detection batch.
            conf: Detection confidence threshold.

        Returns:
            List of YOLO detection results, one per frame.
        """
        detections = []
        total_frames = len(frames)

        for i in range(0, total_frames, batch_size):
            batch = frames[i : i + batch_size]
            try:
                batch_detections = self.model.predict(batch, conf=conf)
                detections += batch_detections
            except Exception as e:
                logger.error(f"Detection failed on batch {i}: {e}")
                # Append empty detections for failed batch
                detections += [None] * len(batch)

        logger.info(f"Detected objects in {total_frames} frames")
        return detections

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Dict[str, List[Dict[int, dict]]]:
        """
        Get tracked object positions across all frames.

        Runs detection, applies ByteTrack tracking, and organizes
        results into a structured dictionary. Supports caching
        results to/from pickle files.

        Args:
            frames: List of BGR video frames.
            read_from_stub: If True, try to load cached results.
            stub_path: Path to the pickle cache file.

        Returns:
            Tracking dictionary with keys 'players', 'referees', 'ball'.
            Each value is a list of per-frame dictionaries mapping
            track IDs to their detection data (bbox, etc.).
        """
        # Try to load cached results
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    tracks = pickle.load(f)
                logger.info(f"Loaded tracks from cache: {stub_path}")
                return tracks
            except (pickle.UnpicklingError, EOFError) as e:
                logger.warning(f"Failed to load cache {stub_path}: {e}")

        detections = self.detect_frames(frames)

        tracks: Dict[str, List[Dict[int, dict]]] = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        for frame_num, detection in enumerate(detections):
            if detection is None:
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                continue

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player class
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Apply ByteTrack tracking
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get("player"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv.get("referee"):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv.get("ball"):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save to cache
        if stub_path is not None:
            try:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)
                logger.info(f"Saved tracks cache: {stub_path}")
            except IOError as e:
                logger.warning(f"Failed to save cache {stub_path}: {e}")

        return tracks

    def draw_ellipse(
        self,
        frame: np.ndarray,
        bbox: list[int],
        color: Tuple[int, int, int],
        track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw an ellipse indicator below a detected object.

        Draws a colored ellipse at the bottom of the bounding box
        and optionally a track ID label.

        Args:
            frame: Video frame to draw on.
            bbox: Bounding box [x1, y1, x2, y2].
            color: BGR color tuple for the ellipse.
            track_id: Optional track ID to display as a label.

        Returns:
            Frame with the ellipse drawn on it.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        if track_id is not None:
            rect_width = 40
            rect_height = 20
            x1_rect = x_center - rect_width // 2
            x2_rect = x_center + rect_width // 2
            y1_rect = (y2 - rect_height // 2) + 15
            y2_rect = (y2 + rect_height // 2) + 15

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(
        self,
        frame: np.ndarray,
        bbox: list[int],
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Draw a triangle indicator above a detected object.

        Used to mark players with ball possession or to indicate
        the ball's position.

        Args:
            frame: Video frame to draw on.
            bbox: Bounding box [x1, y1, x2, y2].
            color: BGR color tuple for the triangle.

        Returns:
            Frame with the triangle drawn on it.
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(
        self,
        frame: np.ndarray,
        frame_num: int,
        team_ball_control: np.ndarray,
    ) -> np.ndarray:
        """
        Draw a ball possession statistics overlay.

        Shows cumulative ball possession percentages for both teams
        in a semi-transparent box in the bottom-right corner.

        Args:
            frame: Video frame to draw on.
            frame_num: Current frame index.
            team_ball_control: Array of team IDs (1 or 2) indicating
                which team had ball control at each frame.

        Returns:
            Frame with ball control overlay.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        control_till_frame = team_ball_control[: frame_num + 1]
        team_1_frames = np.sum(control_till_frame == 1)
        team_2_frames = np.sum(control_till_frame == 2)
        total_frames = team_1_frames + team_2_frames

        if total_frames == 0:
            return frame

        team_1_pct = team_1_frames / total_frames * 100
        team_2_pct = team_2_frames / total_frames * 100

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1_pct:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2_pct:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            3,
        )

        return frame

    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: Dict[str, List[Dict[int, dict]]],
        team_ball_control: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Draw all visual annotations on video frames.

        Renders player ellipses (colored by team), referee ellipses,
        ball triangles, ball possession indicators, and team ball
        control statistics.

        Args:
            video_frames: List of BGR video frames.
            tracks: Tracking dictionary with 'players', 'referees', 'ball'.
            team_ball_control: Array of team IDs per frame.

        Returns:
            List of annotated video frames.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players with team colors
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw ball control statistics
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
