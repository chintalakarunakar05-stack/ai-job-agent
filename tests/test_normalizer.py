"""Unit tests for the job data normalizer."""

from src.collectors.normalizer import (
    clean_html,
    clean_whitespace,
    normalize_employment_type,
    normalize_job_dict,
    normalize_location,
    normalize_url,
)
from src.models import JobCreate


def test_clean_html_strips_tags_and_preserves_structure():
    """Verify HTML tags are removed, bullets formatted, and scripts eliminated."""
    html_input = """
    <div>
        <h2>About the Role</h2>
        <p>We are seeking a <strong>Python Developer</strong> to join our team.</p>
        <script>alert('malicious')</script>
        <ul>
            <li>Build REST APIs</li>
            <li>Write unit tests</li>
        </ul>
    </div>
    """
    cleaned = clean_html(html_input)
    assert "<script>" not in cleaned
    assert "alert" not in cleaned
    assert "<h2>" not in cleaned
    assert "About the Role" in cleaned
    assert "Python Developer" in cleaned
    assert "• Build REST APIs" in cleaned
    assert "• Write unit tests" in cleaned


def test_clean_whitespace():
    """Verify redundant spaces, tabs, and multiple blank lines are collapsed."""
    raw = "   Line 1    with    spaces.  \n\n\n\n   Line 2.   \t"
    cleaned = clean_whitespace(raw)
    assert cleaned == "Line 1 with spaces.\n\nLine 2."


def test_normalize_url_strips_tracking_parameters():
    """Verify UTM and tracking query parameters are cleanly stripped from URLs."""
    raw_url = "https://boards.greenhouse.io/example/jobs/12345?gh_src=banner&utm_source=linkedin&utm_medium=job_board#apply"
    norm = normalize_url(raw_url)
    assert "utm_source" not in norm
    assert "gh_src" not in norm
    assert norm == "https://boards.greenhouse.io/example/jobs/12345"


def test_normalize_location_defaults_and_cleans():
    """Verify missing, none, or blank locations default to 'Remote / Unspecified'."""
    assert normalize_location(None) == "Remote / Unspecified"
    assert normalize_location("") == "Remote / Unspecified"
    assert normalize_location("None") == "Remote / Unspecified"
    assert normalize_location("   Hyderabad, India  ") == "Hyderabad, India"


def test_normalize_employment_type():
    """Verify employment terms are categorized cleanly."""
    assert normalize_employment_type("Software Engineer Intern") == "Internship"
    assert normalize_employment_type("Co-op position") == "Internship"
    assert normalize_employment_type("Independent Contractor") == "Contract"
    assert normalize_employment_type("Part-time associate") == "Part-time"
    assert normalize_employment_type("Permanent Regular") == "Full-time"


def test_normalize_job_dict_creates_valid_jobcreate():
    """Verify a complete raw dictionary maps to a validated JobCreate model."""
    raw = {
        "title": "  AI / ML Intern   ",
        "company": "DeepTech Labs",
        "location": "Bengaluru",
        "description": "<p>Build machine learning models with PyTorch.</p>",
        "url": "https://careers.deeptech.com/jobs/99?ref=indeed/",
        "source": "greenhouse",
        "source_job_id": "99",
        "employment_type": "Summer Internship",
        "salary": "₹40,000 / month",
        "posted_date": "2026-03-01T10:00:00Z",
    }
    job = normalize_job_dict(raw)
    assert isinstance(job, JobCreate)
    assert job.title == "AI / ML Intern"
    assert job.company == "DeepTech Labs"
    assert job.employment_type == "Internship"
    assert job.url == "https://careers.deeptech.com/jobs/99"
    assert "Build machine learning models with PyTorch." in job.description
