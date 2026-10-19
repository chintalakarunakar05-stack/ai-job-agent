"""Collector Manager for orchestrating job discovery, normalization, deduplication, and persistence.

Supports command line execution with optional --dry-run flag.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.deduplicator import JobDeduplicator
from src.collectors.greenhouse import GreenhouseCollector
from src.collectors.lever import LeverCollector
from src.config import settings
from src.database import get_db_session, init_db
from src.models import ApplicationStatus, JobCreate, JobModel

logger = logging.getLogger("collector_manager")


class CollectorManager:
    """Orchestrates job discovery from multiple ATS platforms and handles deduplication and storage."""

    def __init__(self, sources_path: Optional[Path] = None):
        """Initialize with path to sources JSON configuration."""
        self.sources_path = sources_path or (settings.DATA_DIR / "sources.json")
        self.greenhouse_collector = GreenhouseCollector()
        self.lever_collector = LeverCollector()

    def load_sources(self) -> Dict[str, List[Dict[str, str]]]:
        """Load configured company boards from JSON.

        Returns:
            Dictionary mapping source platform to list of company board configs.
        """
        if not self.sources_path.exists():
            logger.warning(f"Sources file not found at '{self.sources_path}'. Creating default template.")
            default_sources = {
                "greenhouse": [
                    {"company": "GitLab", "board": "gitlab"},
                    {"company": "Cloudflare", "board": "cloudflare"},
                ],
                "lever": [
                    {"company": "Canonical", "board": "canonical"},
                ],
            }
            with open(self.sources_path, "w", encoding="utf-8") as f:
                json.dump(default_sources, f, indent=2)
            return default_sources

        try:
            with open(self.sources_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read sources config from '{self.sources_path}': {exc}")
            return {"greenhouse": [], "lever": []}

    def collect_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Execute a full collection cycle across all configured platforms.

        Args:
            dry_run: When True, discovers and deduplicates but does NOT write to SQLite.

        Returns:
            Dictionary containing detailed metrics and statistics.
        """
        logger.info(f"Starting job collection cycle (dry_run={dry_run})...")
        sources = self.load_sources()

        stats: Dict[str, Any] = {
            "greenhouse": {"retrieved": 0, "valid": 0, "duplicates": 0},
            "lever": {"retrieved": 0, "valid": 0, "duplicates": 0},
            "total_new_jobs": 0,
            "dry_run": dry_run,
        }

        # Step 1: Pre-load existing jobs from SQLite to prevent database duplication
        init_db()
        with get_db_session() as session:
            existing_jobs = session.query(JobModel).all()
            deduplicator = JobDeduplicator(existing_jobs=existing_jobs)

        all_collected_jobs: List[JobCreate] = []

        # Step 2: Ingest from Greenhouse boards
        for entry in sources.get("greenhouse", []):
            company = entry.get("company", "")
            board = entry.get("board", "")
            if not board:
                continue

            try:
                jobs = self.greenhouse_collector.fetch_jobs(company=company, board=board)
                stats["greenhouse"]["retrieved"] += len(jobs)
                for job in jobs:
                    is_dup, _ = deduplicator.check_duplicate(job)
                    if is_dup:
                        stats["greenhouse"]["duplicates"] += 1
                    else:
                        deduplicator.register(job)
                        all_collected_jobs.append(job)
                        stats["greenhouse"]["valid"] += 1
            except Exception as err:
                logger.error(f"Error collecting Greenhouse for {company}: {err}")

        # Step 3: Ingest from Lever boards
        for entry in sources.get("lever", []):
            company = entry.get("company", "")
            board = entry.get("board", "")
            if not board:
                continue

            try:
                jobs = self.lever_collector.fetch_jobs(company=company, board=board)
                stats["lever"]["retrieved"] += len(jobs)
                for job in jobs:
                    is_dup, _ = deduplicator.check_duplicate(job)
                    if is_dup:
                        stats["lever"]["duplicates"] += 1
                    else:
                        deduplicator.register(job)
                        all_collected_jobs.append(job)
                        stats["lever"]["valid"] += 1
            except Exception as err:
                logger.error(f"Error collecting Lever for {company}: {err}")

        stats["total_new_jobs"] = len(all_collected_jobs)

        # Step 4: Save to Database if not dry_run
        if not dry_run and all_collected_jobs:
            inserted_count = self._persist_jobs(all_collected_jobs)
            logger.info(f"Successfully persisted {inserted_count} new opportunities to database.")
        elif dry_run:
            logger.info(f"[DRY-RUN] Staged {len(all_collected_jobs)} new opportunities (database not modified).")

        return stats

    def _persist_jobs(self, jobs: List[JobCreate]) -> int:
        """Insert verified non-duplicate jobs into SQLite."""
        inserted = 0
        with get_db_session() as session:
            for item in jobs:
                # Double-check against database constraint before insert
                existing = session.query(JobModel).filter_by(url=item.url).first()
                if existing:
                    continue

                db_job = JobModel(
                    title=item.title,
                    company=item.company,
                    location=item.location,
                    description=item.description,
                    url=item.url,
                    source=item.source,
                    source_job_id=item.source_job_id,
                    employment_type=item.employment_type,
                    salary=item.salary,
                    posted_date=item.posted_date,
                    application_url=item.application_url,
                    status=ApplicationStatus.NEW,
                )
                session.add(db_job)
                inserted += 1

        return inserted


def print_stats(stats: Dict[str, Any]) -> None:
    """Print readable summary of the collection cycle."""
    mode_label = "[DRY-RUN MODE]" if stats.get("dry_run") else "[LIVE PERSISTENCE MODE]"
    print("\n" + "=" * 55)
    print(f"       JOB COLLECTION REPORT {mode_label}")
    print("=" * 55)

    print("\nGreenhouse:")
    print(f"  Retrieved:  {stats['greenhouse']['retrieved']}")
    print(f"  Valid:      {stats['greenhouse']['valid']}")
    print(f"  Duplicates: {stats['greenhouse']['duplicates']}")

    print("\nLever:")
    print(f"  Retrieved:  {stats['lever']['retrieved']}")
    print(f"  Valid:      {stats['lever']['valid']}")
    print(f"  Duplicates: {stats['lever']['duplicates']}")

    print("\n" + "-" * 55)
    print(f"Total New Jobs: {stats['total_new_jobs']}")
    print("=" * 55 + "\n")


def main() -> None:
    """CLI runner entry point."""
    parser = argparse.ArgumentParser(description="AI Job Agent Job Collector CLI")
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Run collection, normalization, and deduplication without saving to database",
    )
    args = parser.parse_args()

    # Configure root logging for CLI run
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    manager = CollectorManager()
    stats = manager.collect_all(dry_run=args.dry_run)
    print_stats(stats)


if __name__ == "__main__":
    main()
