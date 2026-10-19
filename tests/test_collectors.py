"""Unit tests for Greenhouse and Lever collectors using mocked HTTP responses."""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.collectors.greenhouse import GreenhouseCollector
from src.collectors.lever import LeverCollector
from src.collectors.manager import CollectorManager
from src.models import JobModel, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


SAMPLE_GREENHOUSE_PAYLOAD = {
    "jobs": [
        {
            "id": 987654,
            "title": "Junior Python Developer",
            "location": {"name": "Remote, Global"},
            "content": "<p>Build scalable backend systems with Python and FastAPI.</p>",
            "absolute_url": "https://boards.greenhouse.io/gitlab/jobs/987654",
            "updated_at": "2026-03-10T12:00:00Z",
        },
        {
            "id": 987655,
            "title": "Machine Learning Intern",
            "location": {"name": "Bengaluru, India"},
            "content": "<p>Train vision models with PyTorch and OpenCV.</p>",
            "absolute_url": "https://boards.greenhouse.io/gitlab/jobs/987655",
            "updated_at": "2026-03-11T12:00:00Z",
        },
    ]
}

SAMPLE_LEVER_PAYLOAD = [
    {
        "id": "lever-uuid-001",
        "text": "AI Engineer - Core Team",
        "createdAt": 1710100000000,
        "categories": {
            "commitment": "Full time",
            "location": "Hyderabad, India",
            "team": "Engineering",
        },
        "description": "<p>Develop state of the art AI agent frameworks.</p>",
        "lists": [
            {
                "text": "Qualifications",
                "content": "<ul><li>Python 3.11</li><li>PyTorch</li></ul>",
            }
        ],
        "hostedUrl": "https://jobs.lever.co/canonical/lever-uuid-001",
        "applyUrl": "https://jobs.lever.co/canonical/lever-uuid-001/apply",
    }
]


def test_greenhouse_fetch_jobs_success():
    """Verify Greenhouse collector properly parses and normalizes jobs from API."""
    collector = GreenhouseCollector()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_GREENHOUSE_PAYLOAD

    with patch.object(collector.session, "get", return_value=mock_resp):
        jobs = collector.fetch_jobs(company="GitLab", board="gitlab")

    assert len(jobs) == 2
    assert jobs[0].title == "Junior Python Developer"
    assert jobs[0].company == "GitLab"
    assert jobs[0].source == "greenhouse"
    assert jobs[0].source_job_id == "987654"
    assert "FastAPI" in jobs[0].description
    assert jobs[1].title == "Machine Learning Intern"


def test_greenhouse_handles_http_errors_gracefully():
    """Verify Greenhouse collector handles 404 and 500 status codes without crashing."""
    collector = GreenhouseCollector()

    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404

    with patch.object(collector.session, "get", return_value=mock_resp_404):
        jobs = collector.fetch_jobs(company="UnknownCorp", board="nonexistent")
    assert jobs == []

    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    mock_resp_500.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")

    with patch.object(collector.session, "get", return_value=mock_resp_500):
        jobs = collector.fetch_jobs(company="ErrorCorp", board="error")
    assert jobs == []


def test_greenhouse_handles_timeout_gracefully():
    """Verify Greenhouse collector handles network timeouts cleanly."""
    collector = GreenhouseCollector()

    with patch.object(collector.session, "get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        jobs = collector.fetch_jobs(company="TimeoutCorp", board="timeout")
    assert jobs == []


def test_lever_fetch_jobs_success():
    """Verify Lever collector properly parses and normalizes jobs from API."""
    collector = LeverCollector()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_LEVER_PAYLOAD

    with patch.object(collector.session, "get", return_value=mock_resp):
        jobs = collector.fetch_jobs(company="Canonical", board="canonical")

    assert len(jobs) == 1
    assert jobs[0].title == "AI Engineer - Core Team"
    assert jobs[0].company == "Canonical"
    assert jobs[0].source == "lever"
    assert jobs[0].source_job_id == "lever-uuid-001"
    assert "Python 3.11" in jobs[0].description
    assert jobs[0].location == "Hyderabad, India"


def test_lever_handles_network_error_gracefully():
    """Verify Lever collector handles connection errors without crashing."""
    collector = LeverCollector()

    with patch.object(collector.session, "get", side_effect=requests.exceptions.ConnectionError("Network down")):
        jobs = collector.fetch_jobs(company="DownCorp", board="down")
    assert jobs == []


def test_collector_manager_dry_run(tmp_path):
    """Verify CollectorManager processes collectors without modifying database in dry-run mode."""
    sources_file = tmp_path / "test_sources.json"
    sources_file.write_text('{"greenhouse": [{"company": "GitLab", "board": "gitlab"}], "lever": []}')

    manager = CollectorManager(sources_path=sources_file)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_GREENHOUSE_PAYLOAD

    with patch.object(manager.greenhouse_collector.session, "get", return_value=mock_resp):
        stats = manager.collect_all(dry_run=True)

    assert stats["dry_run"] is True
    assert stats["greenhouse"]["retrieved"] == 2
    assert stats["greenhouse"]["valid"] == 2
    assert stats["total_new_jobs"] == 2
