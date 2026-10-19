"""Greenhouse ATS job collector adapter.

Fetches public job postings from Greenhouse Job Board API:
https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true
"""

import logging
from typing import Any, Dict, List, Optional
import requests

from src.collectors.normalizer import normalize_job_dict
from src.models import JobCreate

logger = logging.getLogger(__name__)


class GreenhouseCollector:
    """Collector for publicly accessible Greenhouse job boards."""

    BASE_URL = "https://boards-api.greenhouse.io/v1/boards"

    def __init__(self, timeout: int = 15):
        """Initialize collector with request timeout."""
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Job-Agent/1.0 (Job Opportunity Aggregator)",
            "Accept": "application/json",
        })

    def fetch_jobs(self, company: str, board: str) -> List[JobCreate]:
        """Fetch all public job postings for a given company board.

        Args:
            company: Human-readable company name (e.g. 'GitLab').
            board: Greenhouse board token / slug (e.g. 'gitlab').

        Returns:
            List of normalized JobCreate objects.
        """
        api_url = f"{self.BASE_URL}/{board}/jobs?content=true"
        logger.info(f"Greenhouse: Fetching jobs for '{company}' ({api_url})...")

        try:
            response = self.session.get(api_url, timeout=self.timeout)
            if response.status_code == 404:
                logger.warning(f"Greenhouse: Board '{board}' for company '{company}' not found (404).")
                return []
            response.raise_for_status()

            data: Dict[str, Any] = response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Greenhouse: Request timed out for '{company}' (board: {board})")
            return []
        except requests.exceptions.RequestException as exc:
            logger.error(f"Greenhouse: Network error for '{company}' (board: {board}): {exc}")
            return []
        except ValueError as exc:
            logger.error(f"Greenhouse: Invalid JSON received for '{company}' (board: {board}): {exc}")
            return []
        except Exception as exc:
            logger.error(f"Greenhouse: Unexpected error fetching '{company}' (board: {board}): {exc}")
            return []

        raw_jobs = data.get("jobs", [])
        if not isinstance(raw_jobs, list):
            logger.warning(f"Greenhouse: 'jobs' field is not a list for '{company}'.")
            return []

        logger.info(f"Greenhouse: Found {len(raw_jobs)} raw postings for '{company}'.")
        normalized_jobs: List[JobCreate] = []

        for item in raw_jobs:
            if not isinstance(item, dict):
                continue

            try:
                # Extract location name safely
                loc = item.get("location")
                location_name = loc.get("name") if isinstance(loc, dict) else str(loc or "")

                raw_dict = {
                    "title": item.get("title"),
                    "company": company,
                    "location": location_name,
                    "description": item.get("content", ""),
                    "url": item.get("absolute_url"),
                    "source": "greenhouse",
                    "source_job_id": str(item.get("id", "")),
                    "employment_type": "Full-time",  # Greenhouse API content includes details in description
                    "posted_date": item.get("updated_at"),
                    "application_url": item.get("absolute_url"),
                }

                # Only keep jobs that have title and URL
                if raw_dict["title"] and raw_dict["url"]:
                    normalized_jobs.append(normalize_job_dict(raw_dict))
            except Exception as err:
                logger.warning(f"Greenhouse: Skipped malformed item in '{company}': {err}")

        logger.info(f"Greenhouse: Successfully normalized {len(normalized_jobs)} jobs for '{company}'.")
        return normalized_jobs
