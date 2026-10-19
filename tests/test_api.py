"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models import ApplicationStatus


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ollama_model" in data


def test_get_profile(client):
    response = client.get("/api/profile")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "email" in data
    assert "target_roles" in data


def test_get_stats(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_jobs" in data
    assert "high_matches" in data


def test_list_jobs(client):
    response = client.get("/api/jobs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "jobs" in data
    assert len(data["jobs"]) <= 5


def test_get_job_detail_and_status_update(client):
    # Fetch first job
    list_res = client.get("/api/jobs?limit=1")
    assert list_res.status_code == 200
    jobs = list_res.json()["jobs"]
    if not jobs:
        pytest.skip("No jobs in database to test detail")

    job_id = jobs[0]["id"]
    detail_res = client.get(f"/api/jobs/{job_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == job_id

    # Test patch status
    patch_res = client.patch(
        f"/api/jobs/{job_id}/status",
        json={"status": ApplicationStatus.SHORTLISTED.value},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["status"] == ApplicationStatus.SHORTLISTED.value

    # Revert back
    client.patch(
        f"/api/jobs/{job_id}/status",
        json={"status": ApplicationStatus.NEW.value},
    )


def test_resume_recommendation_endpoint(client):
    list_res = client.get("/api/jobs?limit=1")
    assert list_res.status_code == 200
    jobs = list_res.json()["jobs"]
    if not jobs:
        pytest.skip("No jobs in database")

    job_id = jobs[0]["id"]
    rec_res = client.get(f"/api/resumes/recommend/{job_id}")
    assert rec_res.status_code == 200
    data = rec_res.json()
    assert "category" in data
    assert "suggested_filename" in data
