"""Lever ATS job collector adapter.

Fetches public job postings from Lever Postings API:
https://api.lever.co/v0/postings/{board}?mode=json
"""

import logging
from typing import Any, Dict, List, Optional
import requests

from src.collectors.normalizer import normalize_job_dict
from src.models import JobCreate

logger = logging.getLogger(__name__)


class LeverCollector:
    """Collector for publicly accessible Lever job boards."""

    BASE_URL = "https://api.lever.co/v0/postings"

    def __init__(self, timeout: int = 15):
        """Initialize collector with request timeout."""
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Job-Agent/1.0 (Job Opportunity Aggregator)",
            "Accept": "application/json",
        })

    def fetch_jobs(self, company: str, board: str) -> List[JobCreate]:
        """Fetch all public job postings for a given company Lever board.

        Args:
            company: Human-readable company name (e.g. 'Canonical').
            board: Lever board token / site slug (e.g. 'canonical').

        Returns:
            List of normalized JobCreate objects.
        """
        api_url = f"{self.BASE_URL}/{board}?mode=json"
        logger.info(f"Lever: Fetching jobs for '{company}' ({api_url})...")

        try:
            response = self.session.get(api_url, timeout=self.timeout)
            if response.status_code == 404:
                logger.warning(f"Lever: Board '{board}' for company '{company}' not found (404).")
                return []
            response.raise_for_status()

            data = response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Lever: Request timed out for '{company}' (board: {board})")
            return []
        except requests.exceptions.RequestException as exc:
            logger.error(f"Lever: Network error for '{company}' (board: {board}): {exc}")
            return []
        except ValueError as exc:
            logger.error(f"Lever: Invalid JSON received for '{company}' (board: {board}): {exc}")
            return []
        except Exception as exc:
            logger.error(f"Lever: Unexpected error fetching '{company}' (board: {board}): {exc}")
            return []

        if not isinstance(data, list):
            logger.warning(f"Lever: Expected list of jobs from API, got {type(data)} for '{company}'.")
            return []

        logger.info(f"Lever: Found {len(data)} raw postings for '{company}'.")
        normalized_jobs: List[JobCreate] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                categories = item.get("categories", {}) or {}
                location = categories.get("location", "")
                commitment = categories.get("commitment", "Full-time")

                # Combine description and lists for a complete description
                desc_parts = []
                if item.get("description"):
                    desc_parts.append(item["description"])
                
                # Lever separates bullet lists in a 'lists' array
                for sublist in item.get("lists", []):
                    header = sublist.get("text", "")
                    content = sublist.get("content", "")
                    if header or content:
                        desc_parts.append(f"<h3>{header}</h3>{content}")

                if item.get("additional"):
                    desc_parts.append(item["additional"])

                full_desc = "\n".join(desc_parts)

                raw_dict = {
                    "title": item.get("text"),
                    "company": company,
                    "location": location,
                    "description": full_desc,
                    "url": item.get("hostedUrl"),
                    "source": "lever",
                    "source_job_id": str(item.get("id", "")),
                    "employment_type": commitment,
                    "posted_date": item.get("createdAt"),
                    "application_url": item.get("applyUrl") or item.get("hostedUrl"),
                }

                if raw_dict["title"] and raw_dict["url"]:
                    normalized_jobs.append(normalize_job_dict(raw_dict))
            except Exception as err:
                logger.warning(f"Lever: Skipped malformed item in '{company}': {err}")

        logger.info(f"Lever: Successfully normalized {len(normalized_jobs)} jobs for '{company}'.")
        return normalized_jobs
