import cv2


def get_video_properties(video_path):
    """
    Get video properties including frame rate, width, height, and frame count.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with 'fps', 'width', 'height', 'frame_count' keys
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {
        'fps': fps if fps > 0 else 24,  # Default to 24 if detection fails
        'width': width,
        'height': height,
        'frame_count': frame_count
    }


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def save_video(ouput_video_frames, output_video_path, fps=24):
    """
    Save video frames to a file.
    
    Args:
        ouput_video_frames: List of video frames
        output_video_path: Path where the video will be saved
        fps: Frame rate for the output video (default: 24)
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]),
    )
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
