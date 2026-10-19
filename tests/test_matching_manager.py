"""Integration tests for the MatchingManager."""

from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.ai.ai_matcher import AIMatchResult, AIMatcher
from src.ai.matching_manager import MatchingManager
from src.models import (
    ApplicationStatus,
    Base,
    EducationItem,
    JobModel,
    SalaryPreference,
    UserProfile,
    WorkPreference,
)


@pytest.fixture
def mock_profile() -> UserProfile:
    return UserProfile(
        name="Candidate Test",
        email="candidate@test.com",
        phone="+91 9999999999",
        location="Hyderabad, India",
        education=[
            EducationItem(
                degree="Bachelor of Technology",
                field_of_study="Computer Science",
                institution="Tech College",
                graduation_year=2026,
            )
        ],
        skills=["Python", "Machine Learning", "PyTorch", "FastAPI"],
        target_roles=["AI Engineer Intern", "Python Developer Intern"],
        preferred_locations=["Hyderabad", "Remote"],
        employment_types=["Internship", "Full-time"],
    )


@pytest.fixture
def test_db_with_jobs(tmp_path, monkeypatch):
    """Create a temporary test database populated with sample jobs."""
    db_file = tmp_path / "test_match.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    # Populate jobs
    with Session() as session:
        jobs = [
            JobModel(
                title="AI Engineer Intern",
                company="Alpha AI",
                location="Hyderabad",
                description="Python, PyTorch, and machine learning required for intern.",
                url="https://alpha.example/jobs/1",
                source="greenhouse",
                status=ApplicationStatus.NEW,
            ),
            JobModel(
                title="Junior Python Developer",
                company="Beta Devs",
                location="Remote",
                description="Python backend development with FastAPI.",
                url="https://beta.example/jobs/2",
                source="lever",
                status=ApplicationStatus.NEW,
            ),
            JobModel(
                title="Vice President of Accounting",
                company="Gamma Corp",
                location="London, UK",
                description="15+ years experience in corporate accounting and audit.",
                url="https://gamma.example/jobs/3",
                source="greenhouse",
                status=ApplicationStatus.NEW,
            ),
        ]
        session.add_all(jobs)
        session.commit()

    # Monkeypatch engine and SessionLocal in src.database
    import src.database as db_mod
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", Session)

    return Session


# ---------------------------------------------------------------------------
# Empty database
# ---------------------------------------------------------------------------

def test_matching_manager_empty_database(mock_profile, monkeypatch):
    """Verify MatchingManager handles an empty database without error."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    import src.database as db_mod
    monkeypatch.setattr(db_mod, "engine", engine)
    monkeypatch.setattr(db_mod, "SessionLocal", Session)

    manager = MatchingManager(profile=mock_profile)
    stats = manager.run_matching(dry_run=True)
    assert stats["total_jobs"] == 0
    assert stats["deterministic_evaluated"] == 0


# ---------------------------------------------------------------------------
# Normal hybrid run (with AI)
# ---------------------------------------------------------------------------

def test_matching_manager_ranks_and_evaluates(mock_profile, test_db_with_jobs):
    """Verify MatchingManager ranks top jobs, evaluates with AI, and updates SQLite."""
    mock_ai = MagicMock(spec=AIMatcher)
    mock_ai.evaluate.return_value = AIMatchResult(
        ai_score=90.0,
        recommendation="EXCELLENT_MATCH",
        reason="Superb technical fit with candidate.",
        matched_requirements=["Python", "PyTorch"],
        missing_requirements=[],
        risk_flags=[],
        eligibility=True,
        success=True,
    )

    manager = MatchingManager(profile=mock_profile, ai_matcher=mock_ai)
    stats = manager.run_matching(ai_limit=2, dry_run=False, min_ai_threshold=50.0)

    assert stats["total_jobs"] == 3
    assert stats["deterministic_evaluated"] == 3
    assert stats["ai_evaluated"] <= 2
    assert stats["ai_succeeded"] <= 2

    # Verify SQLite was updated
    Session = test_db_with_jobs
    with Session() as session:
        top_job = session.query(JobModel).filter_by(title="AI Engineer Intern").first()
        assert top_job.deterministic_score > 70.0
        assert top_job.final_score is not None
        assert top_job.match_category is not None
        assert "Python" in top_job.matched_skills

        # Low matching job should have low score and no AI call
        poor_job = session.query(JobModel).filter_by(title="Vice President of Accounting").first()
        assert poor_job.deterministic_score < 40.0
        assert poor_job.ai_score is None


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def test_matching_manager_dry_run_does_not_modify_db(mock_profile, test_db_with_jobs):
    """Verify dry-run does not write match results to SQLite."""
    mock_ai = MagicMock(spec=AIMatcher)
    mock_ai.evaluate.return_value = AIMatchResult(
        ai_score=85.0,
        recommendation="STRONG_MATCH",
        reason="Good fit",
        success=True,
    )

    manager = MatchingManager(profile=mock_profile, ai_matcher=mock_ai)
    stats = manager.run_matching(ai_limit=2, dry_run=True, min_ai_threshold=50.0)

    assert stats["dry_run"] is True

    # Database records should remain unpopulated in dry-run
    Session = test_db_with_jobs
    with Session() as session:
        job = session.query(JobModel).filter_by(title="AI Engineer Intern").first()
        assert job.final_score is None or job.final_score == 0.0


# ---------------------------------------------------------------------------
# --no-ai mode
# ---------------------------------------------------------------------------

def test_matching_manager_no_ai_mode_makes_no_ai_calls(mock_profile, test_db_with_jobs):
    """Verify --no-ai mode: no AI calls are made and deterministic score is the final score."""
    mock_ai = MagicMock(spec=AIMatcher)

    manager = MatchingManager(profile=mock_profile, ai_matcher=mock_ai)
    stats = manager.run_matching(ai_limit=10, dry_run=False, no_ai=True)

    # No AI calls should have been made
    mock_ai.evaluate.assert_not_called()

    assert stats["ai_evaluated"] == 0
    assert stats["ai_succeeded"] == 0
    assert stats["ai_failed"] == 0
    assert stats["no_ai_mode"] is True
    assert stats["total_jobs"] == 3
    assert stats["deterministic_evaluated"] == 3

    # All category counts should add up to the total
    total_categorized = sum(stats["categories"].values())
    assert total_categorized == 3


def test_matching_manager_no_ai_dry_run_does_not_modify_db(mock_profile, test_db_with_jobs):
    """Verify --no-ai --dry-run: no AI calls, no DB writes."""
    mock_ai = MagicMock(spec=AIMatcher)

    manager = MatchingManager(profile=mock_profile, ai_matcher=mock_ai)
    stats = manager.run_matching(ai_limit=10, dry_run=True, no_ai=True)

    mock_ai.evaluate.assert_not_called()
    assert stats["dry_run"] is True
    assert stats["no_ai_mode"] is True

    Session = test_db_with_jobs
    with Session() as session:
        job = session.query(JobModel).filter_by(title="AI Engineer Intern").first()
        # dry_run=True → database must not have been modified
        assert job.final_score is None or job.final_score == 0.0


def test_matching_manager_ineligible_jobs_get_zero_score(mock_profile, test_db_with_jobs):
    """Verify ineligible jobs (e.g. VP / Senior / Non-tech) receive 0.0 final_score and match_score."""
    manager = MatchingManager(profile=mock_profile)
    stats = manager.run_matching(ai_limit=5, dry_run=False, no_ai=True)

    Session = test_db_with_jobs
    with Session() as session:
        # VP of Accounting is ineligible (leadership + non-engineering)
        vp_job = session.query(JobModel).filter_by(title="Vice President of Accounting").first()
        assert vp_job.final_score == 0.0
        assert vp_job.match_score == 0.0
        assert vp_job.match_category == "POOR_MATCH"
        assert "Ineligible" in vp_job.match_reason

        # AI Engineer Intern is eligible and should retain positive score
        intern_job = session.query(JobModel).filter_by(title="AI Engineer Intern").first()
        assert intern_job.final_score > 70.0
        assert intern_job.match_score > 70.0
