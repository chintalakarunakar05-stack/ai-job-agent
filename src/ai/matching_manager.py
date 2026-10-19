"""Batch Matching Manager coordinating two-stage hybrid matching.
Workflow:
1. Load candidate profile and un-scored jobs from SQLite.
2. Run Stage 1 Deterministic Matcher across all jobs (fast).
3. Rank jobs by preliminary deterministic score.
4. Filter top N candidates above score threshold for Stage 2 AI semantic analysis.
5. Combine scores into weighted hybrid score.
6. Persist scores, matched skills, missing skills, and recommendations to database.
"""
import argparse
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.ai_matcher import AIMatcher, calculate_hybrid_score
from src.ai.matcher import DeterministicMatcher, MatchCategory
from src.config import settings
from src.database import get_db_session, init_db
from src.models import ApplicationStatus, JobModel, UserProfile, load_user_profile

logger = logging.getLogger("matching_manager")


class MatchingManager:
    """Orchestrates hybrid job matching combining deterministic rules and LLM semantic evaluation."""

    def __init__(
        self,
        profile: Optional[UserProfile] = None,
        ai_matcher: Optional[AIMatcher] = None,
    ):
        """Initialize matching manager."""
        self.profile = profile or load_user_profile()
        self.deterministic_matcher = DeterministicMatcher(self.profile)
        self.ai_matcher = ai_matcher or AIMatcher()

    def run_matching(
        self,
        ai_limit: Optional[int] = None,
        dry_run: bool = False,
        min_ai_threshold: float = 60.0,
        no_ai: bool = False,
    ) -> Dict[str, Any]:
        """Execute two-stage hybrid matching."""
        init_db()
        limit = ai_limit if ai_limit is not None else settings.AI_MATCH_LIMIT

        logger.info(
            f"Initiating hybrid matching cycle "
            f"(ai_limit={limit}, dry_run={dry_run}, no_ai={no_ai})..."
        )
        if no_ai:
            logger.info("--no-ai flag active: Stage 2 AI evaluation is DISABLED. Using deterministic scores only.")

        stats: Dict[str, Any] = {
            "total_jobs": 0,
            "deterministic_evaluated": 0,
            "ai_evaluated": 0,
            "ai_succeeded": 0,
            "ai_failed": 0,
            "no_ai_mode": no_ai,
            "eligible": 0,          # NEW
            "ineligible": 0,        # NEW
            "categories": {
                MatchCategory.EXCELLENT_MATCH: 0,
                MatchCategory.STRONG_MATCH: 0,
                MatchCategory.GOOD_MATCH: 0,
                MatchCategory.POSSIBLE_MATCH: 0,
                MatchCategory.WEAK_MATCH: 0,
                MatchCategory.POOR_MATCH: 0,
            },
            "dry_run": dry_run,
        }

        # Step 1: Load all jobs from database
        with get_db_session() as session:
            db_jobs = session.query(JobModel).all()
            job_records = [
                {
                    "id": j.id,
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "description": j.description,
                    "employment_type": j.employment_type,
                    "status": j.status,
                }
                for j in db_jobs
            ]

        stats["total_jobs"] = len(job_records)
        if not job_records:
            logger.warning("No jobs found in database to evaluate.")
            return stats

        logger.info(f"Stage 1: Evaluating {len(job_records)} jobs with deterministic rules...")

        # Step 2: Deterministic scoring for every job
        evaluated_jobs = []
        for job_data in job_records:
            det_result = self.deterministic_matcher.evaluate(
                title=job_data["title"],
                description=job_data["description"],
                location=job_data["location"],
                employment_type=job_data["employment_type"],
            )
            evaluated_jobs.append({
                "job": job_data,
                "det_result": det_result,
            })
            stats["deterministic_evaluated"] += 1

            # NEW: Count eligibility
            if det_result.eligible:
                stats["eligible"] += 1
            else:
                stats["ineligible"] += 1

        # Step 3: Sort jobs: eligible first, then descending by deterministic score
        evaluated_jobs.sort(
            key=lambda x: (x["det_result"].eligible, x["det_result"].score),
            reverse=True,
        )

        updates: List[Dict[str, Any]] = []

        # Step 4 & 5: Stage 2 AI evaluation (skipped entirely when no_ai=True)
        if no_ai:
            logger.info("Stage 2 skipped (--no-ai mode). All jobs will use deterministic scoring only.")
            ai_candidate_ids: set = set()
        else:
            qualifying_for_ai = [
                item for item in evaluated_jobs
                if item["det_result"].eligible and item["det_result"].score >= min_ai_threshold
            ]
            top_ai_candidates = qualifying_for_ai[:limit]
            ai_candidate_ids = {item["job"]["id"] for item in top_ai_candidates}

            logger.info(
                f"Stage 2: Selected top {len(top_ai_candidates)} eligible jobs "
                f"(deterministic score >= {min_ai_threshold}) for AI evaluation."
            )

            for idx, item in enumerate(top_ai_candidates, 1):
                job = item["job"]
                det_res = item["det_result"]
                logger.info(
                    f"[{idx}/{len(top_ai_candidates)}] AI evaluating: "
                    f"{job['company']} - {job['title']} (Det Score: {det_res.score})"
                )
                ai_res = self.ai_matcher.evaluate(
                    profile=self.profile,
                    title=job["title"],
                    company=job["company"],
                    location=job["location"],
                    description=job["description"],
                    employment_type=job["employment_type"],
                )
                stats["ai_evaluated"] += 1
                if ai_res.success:
                    stats["ai_succeeded"] += 1
                else:
                    stats["ai_failed"] += 1

                final_score, category = calculate_hybrid_score(
                    deterministic_score=det_res.score,
                    ai_score=ai_res.ai_score,
                    ai_success=ai_res.success,
                )
                stats["categories"][category] += 1

                updates.append({
                    "id": job["id"],
                    "deterministic_score": det_res.score,
                    "ai_score": ai_res.ai_score if ai_res.success else None,
                    "final_score": final_score,
                    "match_score": final_score,
                    "match_category": category,
                    "matched_skills": json.dumps(det_res.matched_skills),
                    "missing_skills": json.dumps(det_res.missing_skills),
                    "match_reason": ai_res.reason if ai_res.success else f"Deterministic: {det_res.recommendation}",
                    "ai_analysis": json.dumps(ai_res.model_dump()) if ai_res.success else None,
                    "status": ApplicationStatus.MATCHED if final_score >= 80.0 else job["status"],
                    "analyzed_at": datetime.utcnow(),
                })

        # Step 6: Process all jobs not handled by Stage 2 (deterministic-only and ineligible)
        for item in evaluated_jobs:
            job = item["job"]
            if job["id"] in ai_candidate_ids:
                continue

            det_res = item["det_result"]
            if not det_res.eligible:
                # Ineligible roles (senior, sales, etc.) drop out of top rankings
                final_score = 0.0
                category = MatchCategory.POOR_MATCH
                match_reason = f"Ineligible: {det_res.eligibility_reason}"
            else:
                final_score = det_res.score
                category = det_res.match_category
                match_reason = f"Deterministic: {det_res.recommendation}"

            stats["categories"][category] += 1

            updates.append({
                "id": job["id"],
                "deterministic_score": det_res.score,
                "ai_score": None,
                "final_score": final_score,
                "match_score": final_score,
                "match_category": category,
                "matched_skills": json.dumps(det_res.matched_skills),
                "missing_skills": json.dumps(det_res.missing_skills),
                "match_reason": match_reason,
                "ai_analysis": None,
                "status": job["status"],
                "analyzed_at": datetime.utcnow(),
            })

        # Step 7: Persist results to SQLite if not dry-run
        if not dry_run and updates:
            self._save_updates(updates)
            logger.info(f"Successfully saved match evaluations for {len(updates)} jobs.")
        elif dry_run:
            logger.info(f"[DRY-RUN] Evaluated {len(updates)} jobs (database not modified).")

        return stats

    def _save_updates(self, updates: List[Dict[str, Any]]) -> None:
        """Batch update job records in SQLite."""
        with get_db_session() as session:
            for up in updates:
                db_job = session.query(JobModel).filter_by(id=up["id"]).first()
                if db_job:
                    db_job.deterministic_score = up["deterministic_score"]
                    db_job.ai_score = up["ai_score"]
                    db_job.final_score = up["final_score"]
                    db_job.match_score = up["match_score"]
                    db_job.match_category = up["match_category"]
                    db_job.matched_skills = up["matched_skills"]
                    db_job.missing_skills = up["missing_skills"]
                    db_job.match_reason = up["match_reason"]
                    db_job.ai_analysis = up["ai_analysis"]
                    db_job.status = up["status"]
                    db_job.analyzed_at = up["analyzed_at"]


def print_matching_report(stats: Dict[str, Any]) -> None:
    """Print readable summary of the matching cycle."""
    mode_label = "[DRY-RUN MODE]" if stats.get("dry_run") else "[LIVE DATABASE UPDATED]"
    ai_label = "[DETERMINISTIC ONLY — AI DISABLED]" if stats.get("no_ai_mode") else "[HYBRID: DETERMINISTIC + AI]"

    print("\n" + "=" * 60)
    print(f"         HYBRID JOB MATCHING REPORT {mode_label}")
    print(f"         {ai_label}")
    print("=" * 60)
    print(f"\nTotal Jobs Analyzed:       {stats['total_jobs']}")
    print(f"Deterministic Evaluated:   {stats['deterministic_evaluated']}")
    print(f"AI Semantic Evaluated:     {stats['ai_evaluated']}")
    print(f"  AI Succeeded:            {stats['ai_succeeded']}")
    print(f"  AI Fallbacks:            {stats['ai_failed']}")

    # NEW: Eligibility section
    print("\nEligibility Breakdown:")
    print(f"  Eligible             : {stats.get('eligible', 0)}")
    print(f"  Ineligible           : {stats.get('ineligible', 0)}")

    print("\nClassification Breakdown:")
    for cat, count in stats["categories"].items():
        print(f"  {cat.ljust(18)}: {count}")
    print("=" * 60 + "\n")


def main() -> None:
    """CLI runner entry point."""
    parser = argparse.ArgumentParser(description="AI Job Agent Hybrid Matching Engine")
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Run matching without writing changes to the database",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of top jobs to evaluate with AI (defaults to settings.AI_MATCH_LIMIT)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        dest="no_ai",
        help=(
            "Skip Stage 2 AI (Ollama) evaluation entirely. "
            "All jobs will be scored using deterministic rules only. "
            "Useful when Ollama is unavailable or slow."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    manager = MatchingManager()
    stats = manager.run_matching(ai_limit=args.limit, dry_run=args.dry_run, no_ai=args.no_ai)
    print_matching_report(stats)


if __name__ == "__main__":
    main()