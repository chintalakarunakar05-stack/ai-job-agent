"""Unit tests for the multi-signal job deduplicator."""

from src.collectors.deduplicator import JobDeduplicator, generate_fuzzy_key
from src.models import JobCreate, JobModel


def test_generate_fuzzy_key():
    """Verify fuzzy keys ignore punctuation, casing, and corporate suffixes."""
    key1 = generate_fuzzy_key("GitLab Inc.", "Junior Python Dev", "Remote, US")
    key2 = generate_fuzzy_key("gitlab", "junior python dev", "remote us")
    assert key1 == key2


def test_deduplicate_by_source_and_id():
    """Verify priority 1: same source and source_job_id is flagged as duplicate."""
    dedup = JobDeduplicator()

    job1 = JobCreate(
        title="ML Engineer",
        company="Company A",
        location="Remote",
        description="desc",
        url="https://a.com/job1",
        source="greenhouse",
        source_job_id="GH-100",
    )
    job2 = JobCreate(
        title="Different Title",
        company="Company A",
        location="Different Location",
        description="desc",
        url="https://a.com/job2",
        source="greenhouse",
        source_job_id="GH-100",  # Same source ID
    )

    is_dup1, _ = dedup.check_duplicate(job1)
    assert not is_dup1
    dedup.register(job1)

    is_dup2, reason = dedup.check_duplicate(job2)
    assert is_dup2
    assert "source & ID" in reason


def test_deduplicate_by_url():
    """Verify duplicate posting URL is detected."""
    dedup = JobDeduplicator()

    job1 = JobCreate(
        title="Backend Dev",
        company="Company B",
        location="Hyderabad",
        description="desc",
        url="https://b.com/careers/backend-1",
        source="custom",
    )
    job2 = JobCreate(
        title="Senior Backend Dev",
        company="Company B",
        location="Hyderabad",
        description="desc",
        url="https://b.com/careers/backend-1/",  # Same URL with trailing slash
        source="custom",
    )

    dedup.register(job1)
    is_dup, reason = dedup.check_duplicate(job2)
    assert is_dup
    assert "posting URL" in reason


def test_deduplicate_by_fuzzy_company_title_location():
    """Verify priority 4: company + title + location catches reposted jobs with different URLs."""
    dedup = JobDeduplicator()

    job1 = JobCreate(
        title="Data Science Intern",
        company="Stripe Inc.",
        location="Bengaluru, India",
        description="desc",
        url="https://stripe.com/jobs/1",
        source="greenhouse",
        source_job_id="1",
    )
    job2 = JobCreate(
        title="Data Science Intern",
        company="Stripe",
        location="Bengaluru India",
        description="desc",
        url="https://indeed.com/viewjob/stripe-2",  # Different URL and source
        source="indeed",
        source_job_id="2",
    )

    dedup.register(job1)
    is_dup, reason = dedup.check_duplicate(job2)
    assert is_dup
    assert "company + title + location" in reason


def test_deduplicate_against_existing_db_records():
    """Verify preloaded database records prevent inserting duplicates."""
    existing_db_job = JobModel(
        id=1,
        source="lever",
        source_job_id="LEV-999",
        title="Cloud Architect",
        company="CloudCorp",
        location="Remote",
        description="desc",
        url="https://jobs.lever.co/cloudcorp/LEV-999",
    )

    dedup = JobDeduplicator(existing_jobs=[existing_db_job])

    new_incoming_job = JobCreate(
        title="Cloud Architect",
        company="CloudCorp",
        location="Remote",
        description="desc",
        url="https://jobs.lever.co/cloudcorp/LEV-999",
        source="lever",
        source_job_id="LEV-999",
    )

    is_dup, reason = dedup.check_duplicate(new_incoming_job)
    assert is_dup
    assert "LEV-999" in reason


def test_filter_duplicates_batch():
    """Verify filter_duplicates separates unique items from duplicates in a list."""
    dedup = JobDeduplicator()

    jobs = [
        JobCreate(
            title="Python Dev",
            company="Startup",
            location="Remote",
            description="desc",
            url="https://startup.com/1",
            source="custom",
            source_job_id="1",
        ),
        JobCreate(
            title="Python Dev",
            company="Startup",
            location="Remote",
            description="desc",
            url="https://startup.com/1",  # Exact duplicate
            source="custom",
            source_job_id="1",
        ),
        JobCreate(
            title="React Dev",
            company="Startup",
            location="Remote",
            description="desc",
            url="https://startup.com/2",
            source="custom",
            source_job_id="2",
        ),
    ]

    unique, dup_info = dedup.filter_duplicates(jobs)
    assert len(unique) == 2
    assert len(dup_info) == 1
    assert dup_info[0]["title"] == "Python Dev"
