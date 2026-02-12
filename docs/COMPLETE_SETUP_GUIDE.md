# Complete Setup & Deployment Guide

## SportsVision-AI & RAGFlow

**Author:** Malav Patel  
**Email:** malav.patel203@gmail.com  
**Date:** February 2026  
**Version:** 1.0.0

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [SportsVision-AI Setup](#2-sportsvision-ai-setup)
3. [RAGFlow Setup](#3-ragflow-setup)
4. [GitHub Upload Guide](#4-github-upload-guide)
5. [Docker Deployment](#5-docker-deployment)
6. [Kubernetes Deployment](#6-kubernetes-deployment)
7. [CI/CD Pipeline](#7-cicd-pipeline)
8. [Production Best Practices](#8-production-best-practices)
9. [Future Development Roadmap](#9-future-development-roadmap)
10. [AI Tools for Building Similar Projects](#10-ai-tools-for-building-similar-projects)
11. [Troubleshooting](#11-troubleshooting)
12. [Resources](#12-resources)

---

## 1. Introduction

This guide provides step-by-step instructions for setting up, deploying, and maintaining two production-grade AI projects:

1. **SportsVision-AI**: Real-time sports analytics using computer vision
2. **RAGFlow**: Document Q&A system with conversational memory

Both projects follow industry best practices including:
- Modular architecture
- Docker containerization
- CI/CD automation
- Comprehensive documentation
- Production-ready configurations

---

## 2. SportsVision-AI Setup

### 2.1 Prerequisites

```bash
# System Requirements
- Python 3.11 or higher
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM
- 50GB+ disk space (for models and videos)

# Software Requirements
- Git
- Docker (optional)
- CUDA Toolkit 12.x (for GPU support)
```

### 2.2 Local Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/mlvpatel/SportsVision-AI.git
cd SportsVision-AI

# Step 2: Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Step 3: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Download YOLO model (run this script)
python scripts/download_models.py

# OR manually download from Ultralytics
# Place the model file in models/best.pt
```

### 2.3 Running the Analysis

```bash
# Basic usage with sample video
python main.py --input data/input_videos/match.mp4 \
               --output data/output_videos/output.avi

# With GPU acceleration
python main.py --input match.mp4 --output output/ --device cuda

# Use cached tracking data (faster re-runs)
python main.py --input match.mp4 --use-stubs

# Export statistics to JSON
python main.py --input match.mp4 --export-stats results.json
```

### 2.4 Starting the API Server

```bash
# Start FastAPI server
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation at:
# http://localhost:8000/docs
```

### 2.5 Starting Streamlit Demo

```bash
# In a new terminal
streamlit run app.py --server.port 8501

# Access at: http://localhost:8501
```

### 2.6 Expected Output

After running the analysis, you should see:
```
==============================================================
                 SPORTSVISION-AI RESULTS
==============================================================
  Team 1 Possession: 54.2%
  Team 2 Possession: 45.8%
  Total Frames:      1500
  Processing Time:   45.23s
  Average FPS:       33.2
==============================================================
```

---

## 3. RAGFlow Setup

### 3.1 Prerequisites

```bash
# System Requirements
- Python 3.11 or higher
- 4GB+ RAM
- 10GB+ disk space

# API Keys Required
- OpenAI API Key (required)
- LangSmith API Key (optional, for tracing)
```

### 3.2 Local Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/mlvpatel/RAGFlow.git
cd RAGFlow

# Step 2: Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Step 3: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Configure environment
cp .env.example .env

# Step 5: Edit .env file and add your API keys
# OPENAI_API_KEY=sk-your-key-here
```

### 3.3 Running the Application

```bash
# Terminal 1: Start API server
cd src/api
uvicorn main:app --reload --port 8000

# Terminal 2: Start Streamlit UI
cd frontend
streamlit run streamlit_app.py --server.port 8501
```

### 3.4 Using the Application

1. Open http://localhost:8501 in your browser
2. Use the sidebar to upload a document (PDF, DOCX, TXT)
3. Wait for "Document indexed successfully" message
4. Type your question in the chat input
5. Ask follow-up questions (context is remembered)

### 3.5 API Usage Examples

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-doc",
        files={"file": f}
    )
print(response.json())

# Chat with documents
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What is the main topic?",
        "session_id": "user123"
    }
)
print(response.json()["answer"])
```

---

## 4. GitHub Upload Guide

### 4.1 Create GitHub Repository

```bash
# Step 1: Go to github.com and create a new repository
# Name: SportsVision-AI (or RAGFlow)
# Do NOT initialize with README

# Step 2: Initialize local git
cd SportsVision-AI  # or RAGFlow
git init

# Step 3: Add files
git add .

# Step 4: Create initial commit
git commit -m "Initial commit: SportsVision-AI v1.0.0"

# Step 5: Add remote
git remote add origin https://github.com/mlvpatel/SportsVision-AI.git

# Step 6: Push to GitHub
git branch -M main
git push -u origin main
```

### 4.2 Repository Settings

After pushing:
1. Go to repository Settings
2. Add topics: `computer-vision`, `yolo`, `sports-analytics`, `python`
3. Enable GitHub Actions (Settings > Actions)
4. Add secrets for CI/CD:
   - `OPENAI_API_KEY` (for RAGFlow)
   - `DOCKER_USERNAME` / `DOCKER_PASSWORD` (for Docker Hub)

### 4.3 Creating Releases

```bash
# Create a tag
git tag -a v1.0.0 -m "Release v1.0.0"

# Push tags
git push origin v1.0.0

# Or use GitHub UI:
# Releases > Create new release > Tag: v1.0.0
```

---

## 5. Docker Deployment

### 5.1 Building Docker Images

```bash
# SportsVision-AI
cd SportsVision-AI
docker build -t sportsvision-ai:latest -f docker/Dockerfile .

# RAGFlow
cd RAGFlow
docker build -t ragflow:latest -f docker/Dockerfile .
```

### 5.2 Running with Docker

```bash
# SportsVision-AI (with GPU)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  sportsvision-ai:latest

# RAGFlow
docker run -p 8000:8000 -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  ragflow:latest
```

### 5.3 Docker Compose

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 6. Kubernetes Deployment

### 6.1 Prerequisites

```bash
# Install kubectl
# Install a Kubernetes cluster (minikube, kind, or cloud)
kubectl version --client
```

### 6.2 Deployment Files

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sportsvision-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sportsvision-api
  template:
    metadata:
      labels:
        app: sportsvision-api
    spec:
      containers:
      - name: api
        image: sportsvision-ai:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: sportsvision-service
spec:
  selector:
    app: sportsvision-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 6.3 Deploying to Kubernetes

```bash
# Apply configuration
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/sportsvision-api

# Scale
kubectl scale deployment sportsvision-api --replicas=3
```

---

## 7. CI/CD Pipeline

### 7.1 GitHub Actions Workflow

The CI/CD pipeline automatically:
1. **Lint**: Checks code quality with Black, Flake8, isort
2. **Test**: Runs pytest with coverage
3. **Build**: Creates Docker image
4. **Deploy**: Pushes to registry (on main branch)

### 7.2 Triggering Builds

```bash
# Automatic triggers:
- Push to main/develop branches
- Pull requests to main
- Release creation

# Manual trigger via GitHub UI:
# Actions > Select workflow > Run workflow
```

### 7.3 Adding Secrets

Go to Settings > Secrets and add:
- `OPENAI_API_KEY`
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `AWS_ACCESS_KEY_ID` (for AWS deployment)
- `AWS_SECRET_ACCESS_KEY`

---

## 8. Production Best Practices

### 8.1 Security

```bash
# Use environment variables for secrets
# Never commit .env files
# Use HTTPS in production
# Implement rate limiting
# Add authentication for APIs
```

### 8.2 Monitoring

```bash
# Prometheus metrics
# Grafana dashboards
# Log aggregation (ELK stack)
# Health check endpoints
```

### 8.3 Scaling

```bash
# Horizontal scaling with Kubernetes
# Load balancing
# Caching with Redis
# CDN for static assets
```

### 8.4 Backup

```bash
# Database backups
# Vector store exports
# Model versioning
# Configuration management
```

---

## 9. Future Development Roadmap

### SportsVision-AI

| Version | Features | Timeline |
|---------|----------|----------|
| v1.5 | Real-time streaming, Multi-camera | Q2 2026 |
| v2.0 | Multi-sport support, Pose estimation | Q4 2026 |
| v2.5 | Predictive analytics, Mobile app | Q2 2027 |

### RAGFlow

| Version | Features | Timeline |
|---------|----------|----------|
| v1.5 | Multi-modal support, Authentication | Q2 2026 |
| v2.0 | Knowledge graphs, Multi-tenant | Q4 2026 |
| v2.5 | Custom embeddings, Analytics | Q2 2027 |

---

## 10. AI Tools for Building Similar Projects

### 10.1 For Students & Researchers

#### Computer Vision Projects

| Tool | Purpose | Learning Resource |
|------|---------|-------------------|
| **Ultralytics YOLO** | Object detection | [Docs](https://docs.ultralytics.com/) |
| **Roboflow** | Dataset annotation | [Universe](https://universe.roboflow.com/) |
| **OpenCV** | Image processing | [Tutorials](https://docs.opencv.org/) |
| **Supervision** | Detection utilities | [GitHub](https://github.com/roboflow/supervision) |

#### RAG/LLM Projects

| Tool | Purpose | Learning Resource |
|------|---------|-------------------|
| **LangChain** | LLM orchestration | [Docs](https://python.langchain.com/) |
| **LlamaIndex** | Document indexing | [Docs](https://docs.llamaindex.ai/) |
| **ChromaDB** | Vector storage | [Docs](https://docs.trychroma.com/) |
| **Hugging Face** | Models & datasets | [Hub](https://huggingface.co/) |

### 10.2 Project Ideas for Students

#### Beginner Level
1. Face detection attendance system
2. Simple document Q&A bot
3. Image classification app

#### Intermediate Level
1. Multi-object tracking for traffic
2. Conversational chatbot with memory
3. Real-time video analytics

#### Advanced Level
1. Multi-camera sports analysis
2. Knowledge graph RAG system
3. Edge AI deployment

### 10.3 Free Datasets

| Dataset | Domain | Link |
|---------|--------|------|
| SoccerNet | Sports | [soccernet.org](https://www.soccer-net.org/) |
| COCO | Object Detection | [cocodataset.org](https://cocodataset.org/) |
| MS MARCO | QA/RAG | [microsoft.com](https://microsoft.github.io/msmarco/) |
| Kaggle | Various | [kaggle.com](https://www.kaggle.com/datasets) |

### 10.4 Free Compute Resources

| Platform | GPU | Hours/Month |
|----------|-----|-------------|
| Google Colab | T4/V100 | ~12 hrs/day |
| Kaggle | P100/T4 | 30 hrs/week |
| Lightning AI | T4 | 22 hrs/month |
| Paperspace | Free tier | Limited |

---

## 11. Troubleshooting

### Common Issues

#### SportsVision-AI

```bash
# Issue: CUDA out of memory
# Solution: Reduce batch size or use smaller model
python main.py --model yolov8n.pt  # Use nano model

# Issue: OpenCV import error
# Solution: Install correct version
pip install opencv-python-headless

# Issue: Model not found
# Solution: Download model first
python scripts/download_models.py
```

#### RAGFlow

```bash
# Issue: OpenAI API error
# Solution: Check API key in .env
echo $OPENAI_API_KEY

# Issue: ChromaDB persistence error
# Solution: Ensure directory exists
mkdir -p data/chroma_db

# Issue: Document parsing error
# Solution: Install system dependencies
apt-get install libmagic1  # Linux
brew install libmagic       # Mac
```

---

## 12. Resources

### Documentation
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Tutorials
- [Krish Naik YouTube - LangChain](https://www.youtube.com/@krishnaik06)
- [CampusX - Machine Learning](https://www.youtube.com/@CampusX-official)
- [CodeWithHarry - Python](https://www.youtube.com/@CodeWithHarry)

### Communities
- [LangChain Discord](https://discord.gg/langchain)
- [Ultralytics Discord](https://discord.com/invite/ultralytics)
- [Hugging Face Forums](https://discuss.huggingface.co/)

---

## Contact

**Malav Patel**
- GitHub: [@mlvpatel](https://github.com/mlvpatel)
- LinkedIn: [malavpatel112](https://linkedin.com/in/malavpatel112)
- Email: malav.patel203@gmail.com

---

*Last Updated: February 2026*
