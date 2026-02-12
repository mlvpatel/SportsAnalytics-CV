"""
SportsAnalytics-CV: Real-Time Sports Analytics with Computer Vision

Main entry point for video analysis pipeline.

Author: Malav Patel
Email: malav.patel203@gmail.com
License: MIT
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import json

import cv2
import numpy as np

from src.utils import get_video_properties, read_video, save_video
from src.trackers import Tracker
from src.team_assigner import TeamAssigner
from src.player_ball_assigner import PlayerBallAssigner
from src.camera_movement import CameraMovementEstimator
from src.view_transformer import ViewTransformer
from src.speed_distance import SpeedAndDistance_Estimator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sportsanalytics.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    team1_possession: float
    team2_possession: float
    total_frames: int
    processing_time: float
    fps: float
    player_stats: dict
    
    def to_dict(self) -> dict:
        return {
            'team1_possession': round(self.team1_possession, 2),
            'team2_possession': round(self.team2_possession, 2),
            'total_frames': self.total_frames,
            'processing_time': round(self.processing_time, 2),
            'fps': round(self.fps, 2),
            'player_stats': self.player_stats
        }
    
    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SportsAnalyzer:
    """
    Main class for sports video analysis.
    
    Performs object tracking, team classification, ball possession analysis,
    speed estimation, and generates annotated output videos.
    
    Example:
        >>> analyzer = SportsAnalyzer(model_path="models/best.pt")
        >>> result = analyzer.analyze("match.mp4", "output.mp4")
        >>> print(f"Team 1: {result.team1_possession}%")
    """
    
    def __init__(
        self,
        model_path: str = "models/best.pt",
        device: str = "cuda",
        conf_threshold: float = 0.5
    ):
        """
        Initialize the sports analyzer.
        
        Args:
            model_path: Path to YOLO model weights
            device: Computing device ('cuda' or 'cpu')
            conf_threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        
        logger.info(f"Initializing SportsAnalyzer")
        logger.info(f"Model: {model_path}, Device: {device}")
        
        self.tracker = Tracker(model_path)
        
    def analyze(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        use_stubs: bool = False,
        stub_dir: str = "stubs",
        export_stats: str = None
    ) -> AnalysisResult:
        """
        Analyze a video file.
        
        Args:
            video_path: Input video path
            output_path: Output video path (optional)
            use_stubs: Use cached tracking data
            stub_dir: Directory for stub files
            export_stats: Path to export statistics JSON
            
        Returns:
            AnalysisResult object with statistics
        """
        start_time = time.time()
        logger.info(f"Starting analysis: {video_path}")
        
        # Get video properties (frame rate, dimensions, etc.)
        video_props = get_video_properties(video_path)
        video_fps = video_props['fps']
        logger.info(f"Video properties: {video_props['width']}x{video_props['height']} @ {video_fps:.2f} fps")
        
        # Read video
        video_frames = read_video(video_path)
        total_frames = len(video_frames)
        logger.info(f"Loaded {total_frames} frames")
        
        # Get object tracks
        stub_path = Path(stub_dir) / "track_stubs.pkl"
        tracks = self.tracker.get_object_tracks(
            video_frames,
            read_from_stub=use_stubs and stub_path.exists(),
            stub_path=str(stub_path) if use_stubs else None
        )
        self.tracker.add_position_to_tracks(tracks)
        
        # Camera movement compensation
        camera_estimator = CameraMovementEstimator(video_frames[0])
        cam_stub = Path(stub_dir) / "camera_movement_stub.pkl"
        camera_movement = camera_estimator.get_camera_movement(
            video_frames,
            read_from_stub=use_stubs and cam_stub.exists(),
            stub_path=str(cam_stub) if use_stubs else None
        )
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        
        # View transformation
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)
        
        # Interpolate ball positions
        tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
        
        # Speed and distance estimation (using detected frame rate)
        speed_estimator = SpeedAndDistance_Estimator(frame_rate=video_fps)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # Team assignment
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
        
        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track["bbox"], player_id
                )
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = (
                    team_assigner.team_colors[team]
                )
        
        # Ball possession analysis
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        
        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
            
            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(
                    tracks["players"][frame_num][assigned_player]["team"]
                )
            else:
                team_ball_control.append(
                    team_ball_control[-1] if team_ball_control else 1
                )
        
        team_ball_control = np.array(team_ball_control)
        
        # Calculate possession percentages
        team1_possession = np.sum(team_ball_control == 1) / len(team_ball_control) * 100
        team2_possession = np.sum(team_ball_control == 2) / len(team_ball_control) * 100
        
        logger.info(f"Possession - Team 1: {team1_possession:.1f}%, Team 2: {team2_possession:.1f}%")
        
        # Generate output video
        if output_path:
            logger.info("Generating annotated video...")
            output_frames = self.tracker.draw_annotations(
                video_frames, tracks, team_ball_control
            )
            output_frames = camera_estimator.draw_camera_movement(
                output_frames, camera_movement
            )
            speed_estimator.draw_speed_and_distance(output_frames, tracks)
            save_video(output_frames, output_path, fps=video_fps)
            logger.info(f"Output saved: {output_path}")
        
        # Calculate processing metrics
        processing_time = time.time() - start_time
        fps = total_frames / processing_time
        
        result = AnalysisResult(
            team1_possession=team1_possession,
            team2_possession=team2_possession,
            total_frames=total_frames,
            processing_time=processing_time,
            fps=fps,
            player_stats={}
        )
        
        if export_stats:
            result.to_json(export_stats)
            logger.info(f"Statistics exported: {export_stats}")
            
        logger.info(f"Analysis complete: {processing_time:.2f}s ({fps:.1f} FPS)")
        
        return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SportsAnalytics-CV: Real-Time Sports Analytics"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/input_videos/match.mp4",
        help="Input video path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/output_videos/output_video.avi",
        help="Output video path"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/best.pt",
        help="YOLO model path"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computing device"
    )
    parser.add_argument(
        "--use-stubs",
        action="store_true",
        help="Use cached tracking data"
    )
    parser.add_argument(
        "--export-stats",
        type=str,
        default=None,
        help="Export statistics to JSON"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = SportsAnalyzer(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )
    
    result = analyzer.analyze(
        video_path=args.input,
        output_path=args.output,
        use_stubs=args.use_stubs,
        export_stats=args.export_stats
    )
    
    # Print results
    print("\n" + "="*60)
    print("                 SPORTSANALYTICS-CV RESULTS")
    print("="*60)
    print(f"  Team 1 Possession: {result.team1_possession:.1f}%")
    print(f"  Team 2 Possession: {result.team2_possession:.1f}%")
    print(f"  Total Frames:      {result.total_frames}")
    print(f"  Processing Time:   {result.processing_time:.2f}s")
    print(f"  Average FPS:       {result.fps:.1f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
