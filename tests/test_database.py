"""Unit tests for database models, session management, and relationships."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base, JobModel, ApplicationModel, ApplicationStatus


@pytest.fixture
def test_db_session():
    """Create an in-memory SQLite database session for isolated testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_create_job_in_database(test_db_session):
    """Verify that a job can be inserted and queried successfully."""
    job = JobModel(
        title="Python Developer Intern",
        company="Tech Innovators Inc",
        location="Hyderabad",
        description="Seeking an energetic Python intern skilled in backend development.",
        url="https://techinnovators.example/jobs/python-intern-101",
        source="greenhouse",
        source_job_id="GH-101",
        employment_type="Internship",
        status=ApplicationStatus.NEW,
    )
    test_db_session.add(job)
    test_db_session.commit()
    test_db_session.refresh(job)

    assert job.id is not None
    assert job.title == "Python Developer Intern"
    assert job.company == "Tech Innovators Inc"
    assert job.status == ApplicationStatus.NEW


def test_create_application_with_relationship(test_db_session):
    """Verify that an application links correctly to its parent job."""
    job = JobModel(
        title="AI Engineer Intern",
        company="Cognitive Systems",
        location="Remote",
        description="Work with PyTorch and LLMs.",
        url="https://cognitivesystems.example/jobs/ai-intern-202",
        source="lever",
        source_job_id="LEV-202",
    )
    test_db_session.add(job)
    test_db_session.commit()

    application = ApplicationModel(
        job_id=job.id,
        company=job.company,
        role=job.title,
        resume_used="ai_ml/AI_ML_Resume.pdf",
        status=ApplicationStatus.REVIEW_REQUIRED,
        human_approved=False,
    )
    test_db_session.add(application)
    test_db_session.commit()

    # Query parent job and check related applications
    retrieved_job = test_db_session.query(JobModel).filter_by(id=job.id).first()
    assert len(retrieved_job.applications) == 1
    assert retrieved_job.applications[0].role == "AI Engineer Intern"
    assert retrieved_job.applications[0].human_approved is False
