"""
Unit tests for the FastAPI API endpoints.

Uses FastAPI's TestClient for synchronous endpoint testing.

Author: Malav Patel
"""

import pytest
import unittest.mock

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

skip_without_fastapi = pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="Requires fastapi and httpx",
)


@skip_without_fastapi
class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_fields(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "gpu_available" in data
        assert "model_loaded" in data

    def test_health_version_format(self, client):
        data = client.get("/health").json()
        assert isinstance(data["version"], str)
        assert len(data["version"].split(".")) == 3  # semver


@skip_without_fastapi
class TestAnalyzeEndpoint:
    """Tests for the /analyze endpoint."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    @pytest.fixture
    def mock_celery(self):
        with unittest.mock.patch("src.worker.analyze_video_task") as mock_task:
            yield mock_task

    def test_analyze_returns_job_id(self, client, mock_celery):
        # Setup mock
        mock_task_instance = unittest.mock.Mock()
        mock_task_instance.id = "test-job-id"
        mock_celery.delay.return_value = mock_task_instance

        response = client.post(
            "/analyze",
            json={"video_path": "/tmp/test.mp4"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "pending"


@skip_without_fastapi
class TestStatusEndpoint:
    """Tests for the /status/{job_id} endpoint."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    @pytest.fixture
    def mock_async_result(self):
        with unittest.mock.patch("celery.result.AsyncResult") as mock:
            yield mock

    def test_status_pending(self, client, mock_async_result):
        # Setup mock
        mock_instance = unittest.mock.Mock()
        mock_instance.state = "PENDING"
        mock_async_result.return_value = mock_instance

        response = client.get("/status/test-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"

    def test_status_completed(self, client, mock_async_result):
        # Setup mock
        mock_instance = unittest.mock.Mock()
        mock_instance.state = "SUCCESS"
        mock_instance.result = {
            "job_id": "test-id",
            "status": "completed",
            "team1_possession": 50.0,
            "team2_possession": 50.0,
            "total_frames": 100,
            "processing_time_seconds": 10.0,
            "fps": 30.0,
        }
        mock_async_result.return_value = mock_instance

        response = client.get("/status/test-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"]["total_frames"] == 100


@skip_without_fastapi
class TestDeleteJobEndpoint:
    """Tests for the DELETE /jobs/{job_id} endpoint."""

    @pytest.fixture
    def client(self):
        from src.api.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_async_result(self):
        with unittest.mock.patch("celery.result.AsyncResult") as mock:
            yield mock

    def test_delete_running_job(self, client, mock_async_result):
        # Setup mock
        mock_instance = unittest.mock.Mock()
        mock_instance.state = "STARTED"
        mock_async_result.return_value = mock_instance

        response = client.delete("/jobs/test-id")
        assert response.status_code == 200
        mock_instance.revoke.assert_called_once_with(terminate=True)

    def test_delete_completed_job(self, client, mock_async_result):
        # Setup mock
        mock_instance = unittest.mock.Mock()
        mock_instance.state = "SUCCESS"
        mock_async_result.return_value = mock_instance

        response = client.delete("/jobs/test-id")
        assert response.status_code == 200
        mock_instance.forget.assert_called_once()
