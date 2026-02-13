"""
Unit tests for the FastAPI API endpoints.

Uses FastAPI's TestClient for synchronous endpoint testing.

Author: Malav Patel
"""

import pytest

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

    def test_analyze_returns_job_id(self, client):
        response = client.post(
            "/analyze",
            json={"video_path": "/tmp/test.mp4"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["progress"] == 0.0

    def test_analyze_missing_video_path(self, client):
        response = client.post("/analyze", json={})
        assert response.status_code == 422  # Validation error


@skip_without_fastapi
class TestStatusEndpoint:
    """Tests for the /status/{job_id} endpoint."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_status_not_found(self, client):
        response = client.get("/status/nonexistent-id")
        assert response.status_code == 404

    def test_status_after_submit(self, client):
        # Submit a job first
        submit = client.post(
            "/analyze",
            json={"video_path": "/tmp/test.mp4"},
        )
        job_id = submit.json()["job_id"]

        # Check status
        response = client.get(f"/status/{job_id}")
        assert response.status_code == 200
        assert response.json()["job_id"] == job_id


@skip_without_fastapi
class TestDeleteJobEndpoint:
    """Tests for the DELETE /jobs/{job_id} endpoint."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_delete_not_found(self, client):
        response = client.delete("/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_delete_completed_job(self, client):
        from src.api.main import jobs
        from src.api.models import JobStatus, JobStatusResponse

        # Manually insert a completed job
        job_id = "test-delete-job"
        jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=100.0,
            message="Done",
        )

        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert job_id not in jobs

    def test_delete_processing_job_blocked(self, client):
        from src.api.main import jobs
        from src.api.models import JobStatus, JobStatusResponse

        job_id = "test-processing-job"
        jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=50.0,
            message="In progress",
        )

        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 409

        # Clean up
        del jobs[job_id]


@skip_without_fastapi
class TestOpenAPIDocs:
    """Tests for API documentation endpoints."""

    @pytest.fixture
    def client(self):
        from src.api.main import app

        return TestClient(app)

    def test_docs_endpoint(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "SportsAnalytics-CV API"
        assert "/health" in data["paths"]
        assert "/analyze" in data["paths"]
