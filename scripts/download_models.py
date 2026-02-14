#!/usr/bin/env python3
"""
Download pre-trained YOLO models.

Author: Malav Patel
"""

import os
from pathlib import Path

from ultralytics import YOLO


def download_models():
    """Download YOLO models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("Downloading YOLOv8 models...")

    # Download YOLOv8n (nano - fastest)
    model_n = YOLO("yolov8n.pt")
    print("[OK] YOLOv8n downloaded")

    # Download YOLOv8m (medium - balanced)
    model_m = YOLO("yolov8m.pt")
    print("[OK] YOLOv8m downloaded")

    print("\nModels saved in 'models/' directory")
    print("For custom football model, train using the notebook in notebooks/")


if __name__ == "__main__":
    download_models()
