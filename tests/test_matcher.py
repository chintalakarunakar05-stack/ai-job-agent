"""Unit tests for deterministic job matcher."""
import pytest
from src.ai.matcher import DeterministicMatcher, MatchCategory, categorize_score
from src.models import EducationItem, SalaryPreference, UserProfile, WorkPreference


@pytest.fixture
def sample_profile() -> UserProfile:
    """Fixture providing a realistic entry-level candidate profile."""
    return UserProfile(
        name="Candidate Test",
        email="candidate@test.com",
        phone="+91 9999999999",
        location="Hyderabad, India",
        education=[
            EducationItem(
                degree="Bachelor of Technology",
                field_of_study="Computer Science & AI",
                institution="Tech University",
                graduation_year=2026,
            )
        ],
        skills=[
            "Python",
            "Machine Learning",
            "PyTorch",
            "FastAPI",
            "OpenCV",
            "SQL",
            "Git",
            "Playwright",
        ],
        target_roles=[
            "AI Engineer Intern",
            "Machine Learning Intern",
            "Python Developer Intern",
            "Software Engineer Intern",
        ],
        preferred_locations=[
            "Hyderabad",
            "Bengaluru",
            "Remote",
        ],
        employment_types=[
            "Internship",
            "Full-time",
        ],
        salary_preferences=SalaryPreference(
            currency="INR",
            minimum=25000,
        ),
        work_preferences=WorkPreference(
            remote=True,
            hybrid=True,
            on_site=True,
        ),
    )


def test_categorize_score_thresholds():
    """Verify standard score thresholds map to appropriate categories."""
    assert categorize_score(95.0) == MatchCategory.EXCELLENT_MATCH
    assert categorize_score(85.0) == MatchCategory.STRONG_MATCH
    assert categorize_score(75.0) == MatchCategory.GOOD_MATCH
    assert categorize_score(65.0) == MatchCategory.POSSIBLE_MATCH
    assert categorize_score(50.0) == MatchCategory.WEAK_MATCH
    assert categorize_score(25.0) == MatchCategory.POOR_MATCH


def test_exact_role_and_high_skill_match(sample_profile):
    """Verify that an exact title with strong skill match achieves a high score."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="AI Engineer Intern",
        description=(
            "We are looking for an AI Engineer Intern with Python, "
            "PyTorch, and OpenCV experience. Undergraduate degree in "
            "Computer Science preferred."
        ),
        location="Hyderabad",
        employment_type="Internship",
    )
    assert result.score >= 85.0
    assert result.role_score == 25.0
    assert "Python" in result.matched_skills
    assert "PyTorch" in result.matched_skills
    assert result.experience_score == 15.0
    assert result.location_score == 10.0
    assert result.employment_score == 5.0
    assert result.match_category in [
        MatchCategory.EXCELLENT_MATCH,
        MatchCategory.STRONG_MATCH,
    ]


def test_senior_role_penalty(sample_profile):
    """Verify that a Senior or Director role is heavily penalized on experience."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Senior Staff Software Engineer - AI Systems",
        description=(
            "Must have 8+ years of experience leading engineering teams "
            "with Python and Docker."
        ),
        location="Remote",
        employment_type="Full-time",
    )
    assert result.experience_score <= 3.0
    assert (
        "senior" in result.details["experience_reason"].lower()
        or "8+" in result.details["experience_reason"].lower()
    )


def test_location_matching(sample_profile):
    """Verify location matching correctly scores Remote, preferred cities, and distant locations."""
    matcher = DeterministicMatcher(sample_profile)

    # Remote -> 10 points
    res_remote = matcher.evaluate(
        title="Python Intern",
        description="Python skills",
        location="Remote",
        employment_type="Internship",
    )
    assert res_remote.location_score == 10.0

    # Preferred city -> 10 points
    res_bengaluru = matcher.evaluate(
        title="Python Intern",
        description="Python skills",
        location="Bengaluru, Karnataka",
        employment_type="Internship",
    )
    assert res_bengaluru.location_score == 10.0

    # Outside preferred locations -> 2 points
    res_overseas = matcher.evaluate(
        title="Python Intern",
        description="Python skills",
        location="Berlin, Germany",
        employment_type="Internship",
    )
    assert res_overseas.location_score == 2.0


def test_missing_skills_detection(sample_profile):
    """Verify skills listed in job that are missing from candidate are tracked."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Backend Developer",
        description=(
            "Requires strong Python, Docker, Kubernetes, and AWS knowledge."
        ),
        location="Remote",
    )
    assert "Python" in result.matched_skills
    assert any(
        skill.lower() in ["docker", "kubernetes", "aws"]
        for skill in result.missing_skills
    )


def test_score_boundaries(sample_profile):
    """Verify score stays strictly within [0.0, 100.0]."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Chief Financial Officer / VP Finance",
        description=(
            "15+ years in accounting, CPA certification required, "
            "GAAP compliance."
        ),
        location="Tokyo, Japan",
        employment_type="Contract",
    )
    assert 0.0 <= result.score <= 40.0
    assert result.match_category == MatchCategory.POOR_MATCH


def test_senior_role_is_not_good_match_for_entry_level_candidate(sample_profile):
    """Senior roles should not be recommended as good matches for an entry-level candidate."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Senior Software Engineer",
        description=(
            "We are looking for a Senior Software Engineer with Python, "
            "Docker, and Linux. Requires 5+ years of professional experience."
        ),
        location="Remote",
        employment_type="Full-time",
    )
    assert result.experience_score <= 3.0
    assert result.match_category not in [
        MatchCategory.EXCELLENT_MATCH,
        MatchCategory.STRONG_MATCH,
        MatchCategory.GOOD_MATCH,
    ]


def test_entry_level_role_is_good_match_for_entry_level_candidate(sample_profile):
    """Entry-level roles should remain eligible for an entry-level candidate."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Software Engineer Intern",
        description=(
            "Entry-level software engineering internship for students. "
            "Python, Git, SQL, and machine learning knowledge preferred."
        ),
        location="Remote",
        employment_type="Internship",
    )
    assert result.experience_score == 15.0
    assert result.location_score == 10.0
    assert result.employment_score == 5.0


def test_high_experience_requirement_is_not_recommended_for_entry_level_candidate(
    sample_profile,
):
    """Jobs requiring 5+ years should not be strong/good matches for freshers."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Software Engineer - AI Systems",
        description=(
            "We are looking for a Software Engineer with Python, Docker, "
            "and Linux experience. Requires 5+ years of professional "
            "experience."
        ),
        location="Remote",
        employment_type="Full-time",
    )
    assert result.experience_score <= 3.0
    assert result.match_category not in [
        MatchCategory.EXCELLENT_MATCH,
        MatchCategory.STRONG_MATCH,
        MatchCategory.GOOD_MATCH,
    ]


def test_senior_role_with_matching_skills_is_not_good_match(sample_profile):
    """A senior role must not become a good match just because skills overlap."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Senior Software Engineer, Security Factory: Code Security",
        description=(
            "We are looking for a Senior Software Engineer with Python "
            "and Docker. The role requires significant professional "
            "software engineering experience."
        ),
        location="Remote",
        employment_type="Full-time",
    )
    assert "Python" in result.matched_skills
    assert result.experience_score <= 3.0
    assert result.score < 70.0
    assert result.match_category not in [
        MatchCategory.EXCELLENT_MATCH,
        MatchCategory.STRONG_MATCH,
        MatchCategory.GOOD_MATCH,
    ]


# ============================================================
# Phase 3.1 — Eligibility Rules (new tests)
# ============================================================

def test_senior_role_is_ineligible_for_fresher(sample_profile):
    """Senior roles should be ineligible for an entry-level candidate."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Senior Software Engineer",
        description=(
            "Senior engineer responsible for backend systems. "
            "Requires 5+ years of professional experience."
        ),
        location="Remote",
        employment_type="Full-time",
    )
    assert result.eligible is False
    assert "senior" in result.eligibility_reason.lower()


def test_high_experience_requirement_is_ineligible_for_fresher(sample_profile):
    """Jobs requiring 5+ years should be ineligible for a fresher."""
    matcher = DeterministicMatcher(sample_profile)
    result = matcher.evaluate(
        title="Software Engineer",
        description=(
            "Python software engineer role. "
            "Requires 7+ years of professional experience."
        ),
        location="Remote",
        employment_type="Full-time",
    )
    assert result.eligible is False
    assert "experience" in result.eligibility_reason.lower()


def test_sales_and_account_executive_roles_are_ineligible(sample_profile):
    """Sales, Account Executive, and non-engineering roles should be marked ineligible."""
    matcher = DeterministicMatcher(sample_profile)
    sales_titles = [
        "Strategic Account Executive, AI",
        "Technical Sales Representative",
        "Marketing Specialist",
        "Talent Acquisition Recruiter",
        "Customer Success Manager",
    ]
    for title in sales_titles:
        result = matcher.evaluate(
            title=title,
            description="Entry level role for our sales and growth team.",
            location="Remote",
            employment_type="Full-time",
        )
        assert result.eligible is False, f"Expected {title} to be ineligible"
        assert "non-engineering" in result.eligibility_reason.lower() or "sales" in result.eligibility_reason.lower()