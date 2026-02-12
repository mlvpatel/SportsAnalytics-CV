# SportsAnalytics-CV

**Real-Time Sports Analytics with Computer Vision and Deep Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![YOLO v8/v11](https://img.shields.io/badge/YOLO-v8%2Fv11-orange.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF.svg)](.github/workflows/)

SportsAnalytics-CV is a production-grade computer vision system for real-time sports video analysis. Built on YOLO v8/v11 and OpenCV, it performs player tracking, team classification, ball possession analysis, speed estimation, and generates comprehensive match statistics.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Deployment](#deployment)
- [Datasets](#datasets)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

SportsAnalytics-CV transforms raw sports footage into actionable insights through advanced computer vision techniques.

### Use Cases

- **Match Analysis**: Comprehensive post-match statistics and visualizations
- **Performance Tracking**: Player speed, distance covered, and positioning
- **Team Analytics**: Ball possession percentages and team dynamics
- **Broadcasting**: Real-time overlays for live sports coverage
- **Coaching Tools**: Training analysis and tactical insights

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Object Tracking** | Simultaneous tracking of players, ball, and referees using ByteTrack |
| **Team Classification** | Automatic team assignment using K-Means jersey color clustering |
| **Ball Possession** | Frame-by-frame possession tracking with statistics |
| **Speed Estimation** | Real-time player speed calculation (km/h) |
| **Distance Tracking** | Total distance covered per player |
| **Camera Compensation** | Handles panning and zooming via optical flow |
| **View Transformation** | Bird's-eye view conversion for accurate metrics |
| **Real-Time Visualization** | Annotated video output with overlays |
| **REST API** | FastAPI backend for integration |
| **Streamlit Demo** | Interactive web interface |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SportsAnalytics-CV Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Video   │───▶│  YOLO v8/v11 │───▶│  ByteTrack   │───▶│    Team      │   │
│  │  Input   │    │  Detection   │    │  Tracker     │    │  Classifier  │   │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘   │
│                                              │                    │          │
│                                              ▼                    ▼          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Annotated│◀───│   Metrics    │◀───│    Ball      │◀───│   Speed &    │   │
│  │  Output  │    │   Overlay    │    │  Possession  │    │   Distance   │   │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘   │
│                                              ▲                    ▲          │
│                                              │                    │          │
│                         ┌────────────────────┴────────────────────┘          │
│                         │                                                    │
│                  ┌──────┴──────┐              ┌──────────────┐               │
│                  │   Camera    │              │     View     │               │
│                  │  Movement   │              │  Transformer │               │
│                  └─────────────┘              └──────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   VIDEO INPUT                                                               │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────────┐        │
│   │                    DETECTION LAYER                             │        │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │        │
│   │  │   Players   │  │    Ball     │  │  Referees   │            │        │
│   │  │  Detection  │  │  Detection  │  │  Detection  │            │        │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │        │
│   └───────────────────────────────────────────────────────────────┘        │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────────┐        │
│   │                    TRACKING LAYER                              │        │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │        │
│   │  │  ByteTrack  │  │    Ball     │  │   Position  │            │        │
│   │  │   Multi-Obj │  │ Interpolate │  │   History   │            │        │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │        │
│   └───────────────────────────────────────────────────────────────┘        │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────────┐        │
│   │                   ANALYSIS LAYER                               │        │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │        │
│   │  │    Team     │  │    Ball     │  │   Speed &   │            │        │
│   │  │ Assignment  │  │ Possession  │  │  Distance   │            │        │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │        │
│   └───────────────────────────────────────────────────────────────┘        │
│       │                                                                     │
│       ▼                                                                     │
│   ┌───────────────────────────────────────────────────────────────┐        │
│   │                    OUTPUT LAYER                                │        │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │        │
│   │  │  Annotated  │  │    JSON     │  │    API      │            │        │
│   │  │   Video     │  │   Stats     │  │  Response   │            │        │
│   │  └─────────────┘  └─────────────┘  └─────────────┘            │        │
│   └───────────────────────────────────────────────────────────────┘        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/mlvpatel/SportsAnalytics-CV.git
cd SportsAnalytics-CV

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained model
python scripts/download_models.py
```

### Run Analysis

```bash
# Basic usage
python main.py --input data/input_videos/match.mp4 --output data/output_videos/

# With GPU acceleration
python main.py --input match.mp4 --output output/ --device cuda

# Use cached tracking data
python main.py --input match.mp4 --use-stubs
```

---

## Installation

### Method 1: pip

```bash
pip install -r requirements.txt
```

### Method 2: Docker

```bash
docker build -t sportsanalytics-cv:latest -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/data:/app/data sportsanalytics-cv:latest
```

### Method 3: Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up
```

---

## Usage

### Command Line Interface

```bash
# Process video file
python main.py --input video.mp4 --output results/

# Export statistics to JSON
python main.py --input video.mp4 --export-stats stats.json

# With custom confidence threshold
python main.py --input video.mp4 --conf 0.6
```

### Python API

```python
from sportsanalytics import SportsAnalyzer

# Initialize analyzer
analyzer = SportsAnalyzer(
    model_path="models/yolov8x.pt",
    device="cuda"
)

# Process video
result = analyzer.analyze(
    video_path="match.mp4",
    output_path="output.mp4",
    export_stats=True
)

# Access statistics
print(f"Team 1 Possession: {result.team1_possession:.1f}%")
print(f"Team 2 Possession: {result.team2_possession:.1f}%")
```

### Streamlit Demo

```bash
streamlit run app.py --server.port 8501
```

### REST API

```bash
# Start API server
uvicorn src.api.main:app --reload --port 8000

# Upload and analyze video
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@match.mp4"
```

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze uploaded video |
| `/status/{job_id}` | GET | Get analysis status |
| `/results/{job_id}` | GET | Get analysis results |
| `/health` | GET | Health check |

### Response Schema

```json
{
  "job_id": "uuid",
  "status": "completed",
  "results": {
    "team1_possession": 54.2,
    "team2_possession": 45.8,
    "total_frames": 1500,
    "processing_time": 45.2,
    "player_stats": {}
  }
}
```

---

## Performance

### Benchmarks (NVIDIA RTX 3080)

| Resolution | Model | FPS | mAP |
|------------|-------|-----|-----|
| 1080p | YOLOv8n | 120+ | 89.2% |
| 1080p | YOLOv8m | 85 | 92.1% |
| 1080p | YOLOv8x | 45 | 94.5% |
| 4K | YOLOv8x | 25 | 93.8% |

### Recent Performance Improvements

**Version 2.0 Optimizations:**
- ⚡ **2-5x faster ball interpolation** - Replaced pandas with numpy for efficient processing
- 📊 **Automatic FPS detection** - Accurate speed calculations for any video frame rate
- 💾 **30% less memory usage** - Optimized frame handling in annotation pipeline
- 🎯 **Improved camera tracking** - Reduced memory allocations in optical flow processing

See [PERFORMANCE.md](docs/PERFORMANCE.md) for detailed optimization guide.

### Optimization Tips

- Use TensorRT for 2x speedup on NVIDIA GPUs
- Enable half-precision (FP16) for faster inference
- Use `--use-stubs` for repeated analysis of same video
- Process videos at 720p or 1080p for optimal speed/accuracy balance
- Auto-detected frame rate ensures accurate speed calculations (24/30/60 fps)

### Memory Requirements

| Video Resolution | Recommended RAM | Max Video Length |
|------------------|----------------|------------------|
| 720p | 8 GB | ~20 minutes |
| 1080p | 16 GB | ~10 minutes |
| 4K | 64 GB | ~5 minutes |

---

## Deployment

### Docker

```bash
docker build -t sportsanalytics-cv:latest .
docker run --gpus all -p 8000:8000 sportsanalytics-cv:latest
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

### Cloud Platforms

| Platform | Service | Configuration |
|----------|---------|---------------|
| AWS | SageMaker | `configs/aws_sagemaker.yaml` |
| GCP | Vertex AI | `configs/gcp_vertex.yaml` |
| Azure | ML | `configs/azure_ml.yaml` |

---

## Datasets

### Recommended Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| Football Players Detection | Roboflow annotated | [Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) |
| SoccerNet | Large-scale soccer | [SoccerNet](https://www.soccer-net.org/) |
| DFL Bundesliga | Kaggle competition | [Kaggle](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) |

### Custom Training

```bash
python scripts/train.py --data data/custom_dataset --epochs 100 --model yolov8x
```

---

## Project Structure

```
SportsAnalytics-CV/
├── src/                          # Source code
│   ├── trackers/                 # Object tracking (ByteTrack)
│   ├── team_assigner/            # Team classification (K-Means)
│   ├── player_ball_assigner/     # Ball possession logic
│   ├── camera_movement/          # Camera compensation
│   ├── view_transformer/         # Perspective transform
│   ├── speed_distance/           # Metrics calculation
│   ├── utils/                    # Utility functions
│   └── api/                      # FastAPI backend
├── models/                       # Pre-trained models
├── data/                         # Input/output data
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── docker/                       # Docker configuration
├── docs/                         # Documentation
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── .github/workflows/            # CI/CD pipelines
├── main.py                       # Main entry point
├── app.py                        # Streamlit demo
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Roadmap

### v1.0 (Current)
- [x] Multi-object tracking
- [x] Team classification
- [x] Ball possession tracking
- [x] Speed and distance estimation
- [x] Video annotation

### v1.5 (Q2 2026)
- [ ] Real-time streaming (RTSP/WebRTC)
- [ ] Multi-camera support
- [ ] Event detection (goals, fouls)
- [ ] Heat map generation

### v2.0 (Q4 2026)
- [ ] Multi-sport support (basketball, hockey)
- [ ] Pose estimation integration
- [ ] Predictive analytics
- [ ] Mobile deployment

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Author

**Malav Patel**

- GitHub: [@mlvpatel](https://github.com/mlvpatel)
- LinkedIn: [malavpatel112](https://linkedin.com/in/malavpatel112)
- Email: malav.patel203@gmail.com

---

## Citation

```bibtex
@software{sportsanalytics_cv_2025,
  author = {Patel, Malav},
  title = {SportsAnalytics-CV: Real-Time Sports Analytics},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mlvpatel/SportsAnalytics-CV}
}
```

---

**Star this repo if you find it helpful!**
