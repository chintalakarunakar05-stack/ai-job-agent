"""Multi-signal deduplication engine for job postings.

Detects duplicates using hierarchical prioritized checks:
1. Source + Source Job ID (e.g., 'greenhouse:12345')
2. Canonical Application URL
3. Normalized Posting URL
4. Normalized Company + Title + Location composite key
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.models import JobCreate

logger = logging.getLogger(__name__)


def generate_fuzzy_key(company: str, title: str, location: str) -> str:
    """Generate a sanitized alphanumeric composite key for fuzzy similarity matching."""
    def clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    # Strip common corporate suffixes for higher matching accuracy
    clean_company = re.sub(r"\b(inc|corp|corporation|llc|ltd|limited|technologies|systems)\b", "", company.lower())
    return f"{clean(clean_company)}:{clean(title)}:{clean(location)}"


class JobDeduplicator:
    """Hierarchical deduplication tracker that checks across both in-memory batch and existing database records."""

    def __init__(self, existing_jobs: Optional[Iterable[Any]] = None):
        """Initialize deduplicator with optional existing database jobs.

        Args:
            existing_jobs: List of existing JobModel or JobRead records from database.
        """
        self.seen_source_ids: Set[Tuple[str, str]] = set()
        self.seen_application_urls: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.seen_fuzzy_keys: Set[str] = set()

        if existing_jobs:
            self.load_existing(existing_jobs)

    def load_existing(self, existing_jobs: Iterable[Any]) -> None:
        """Pre-populate tracker with records already stored in the database."""
        count = 0
        for job in existing_jobs:
            self._register(
                source=getattr(job, "source", None),
                source_job_id=getattr(job, "source_job_id", None),
                application_url=getattr(job, "application_url", None),
                url=getattr(job, "url", None),
                company=getattr(job, "company", ""),
                title=getattr(job, "title", ""),
                location=getattr(job, "location", ""),
            )
            count += 1
        logger.debug(f"Preloaded {count} existing jobs into deduplication index.")

    def _register(
        self,
        source: Optional[str],
        source_job_id: Optional[str],
        application_url: Optional[str],
        url: Optional[str],
        company: str,
        title: str,
        location: str,
    ) -> None:
        """Internal helper to index signatures."""
        if source and source_job_id:
            self.seen_source_ids.add((source.lower(), str(source_job_id)))

        if application_url:
            self.seen_application_urls.add(application_url.strip().lower().rstrip("/"))

        if url:
            self.seen_urls.add(url.strip().lower().rstrip("/"))

        if company and title:
            fuzzy = generate_fuzzy_key(company, title, location or "")
            self.seen_fuzzy_keys.add(fuzzy)

    def check_duplicate(self, job: JobCreate) -> Tuple[bool, Optional[str]]:
        """Check if job matches any previously seen job signature.

        Returns:
            Tuple of (is_duplicate: bool, reason: str or None).
        """
        # Priority 1: source + source_job_id
        if job.source and job.source_job_id:
            sig = (job.source.lower(), str(job.source_job_id))
            if sig in self.seen_source_ids:
                return True, f"Match by source & ID ({job.source}:{job.source_job_id})"

        # Priority 2: Canonical application URL
        if job.application_url:
            norm_app_url = job.application_url.strip().lower().rstrip("/")
            if norm_app_url in self.seen_application_urls:
                return True, f"Match by application URL ({job.application_url})"

        # Priority 3: Normalized job posting URL
        if job.url:
            norm_url = job.url.strip().lower().rstrip("/")
            if norm_url in self.seen_urls:
                return True, f"Match by posting URL ({job.url})"

        # Priority 4: Normalized company + title + location
        fuzzy = generate_fuzzy_key(job.company, job.title, job.location or "")
        if fuzzy in self.seen_fuzzy_keys:
            return True, f"Match by company + title + location ({job.company} | {job.title} | {job.location})"

        return False, None

    def register(self, job: JobCreate) -> None:
        """Register a unique job into the deduplication index."""
        self._register(
            source=job.source,
            source_job_id=job.source_job_id,
            application_url=job.application_url,
            url=job.url,
            company=job.company,
            title=job.title,
            location=job.location,
        )

    def filter_duplicates(
        self,
        jobs: List[JobCreate],
    ) -> Tuple[List[JobCreate], List[Dict[str, Any]]]:
        """Filter a list of jobs, separating unique jobs from duplicates.

        Args:
            jobs: List of JobCreate instances.

        Returns:
            Tuple of:
            - unique_jobs: List of non-duplicate JobCreate instances.
            - duplicates_info: List of dicts explaining each duplicate dropped.
        """
        unique_jobs: List[JobCreate] = []
        duplicates_info: List[Dict[str, Any]] = []

        for job in jobs:
            is_dup, reason = self.check_duplicate(job)
            if is_dup:
                logger.debug(f"Duplicate dropped: {job.company} - {job.title} [{reason}]")
                duplicates_info.append({
                    "title": job.title,
                    "company": job.company,
                    "url": job.url,
                    "reason": reason,
                })
            else:
                self.register(job)
                unique_jobs.append(job)

        return unique_jobs, duplicates_info
