"""Unit tests for AI Matcher using mocked Ollama responses.

All tests use mocked OllamaClient — no real Ollama server is required.
"""

from unittest.mock import MagicMock
import pytest

from src.ai.ai_matcher import (
    AIMatchResult,
    AIMatcher,
    calculate_hybrid_score,
    extract_json_from_text,
)
from src.ai.matcher import MatchCategory
from src.models import EducationItem, SalaryPreference, UserProfile, WorkPreference


@pytest.fixture
def sample_profile() -> UserProfile:
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
        target_roles=["AI Engineer Intern", "Machine Learning Intern"],
        preferred_locations=["Hyderabad", "Remote"],
        employment_types=["Internship", "Full-time"],
    )


# ---------------------------------------------------------------------------
# extract_json_from_text
# ---------------------------------------------------------------------------

def test_extract_json_from_text():
    """Verify JSON extractor handles plain JSON, markdown codeblocks, and mixed text."""
    # Plain JSON
    plain = '{"ai_score": 85, "recommendation": "STRONG_MATCH"}'
    assert extract_json_from_text(plain)["ai_score"] == 85

    # Markdown fenced codeblock
    markdown = 'Here is your evaluation:\n```json\n{"ai_score": 90, "recommendation": "EXCELLENT_MATCH"}\n```'
    assert extract_json_from_text(markdown)["ai_score"] == 90

    # Text before and after
    mixed = 'Analysis complete. {"ai_score": 75} Thank you.'
    assert extract_json_from_text(mixed)["ai_score"] == 75

    # Invalid
    assert extract_json_from_text("No JSON here!") is None

    # Empty string
    assert extract_json_from_text("") is None


# ---------------------------------------------------------------------------
# AIMatcher.evaluate — success path
# ---------------------------------------------------------------------------

def test_ai_matcher_success_response(sample_profile):
    """Verify AIMatcher parses valid JSON and populates AIMatchResult."""
    mock_client = MagicMock()
    mock_client.generate.return_value = """
    {
        "ai_score": 88,
        "recommendation": "STRONG_MATCH",
        "reason": "Strong alignment with candidate's Python and ML skills.",
        "matched_requirements": ["Python", "PyTorch"],
        "missing_requirements": ["Docker"],
        "risk_flags": [],
        "eligibility": true
    }
    """
    matcher = AIMatcher(client=mock_client)
    result = matcher.evaluate(
        profile=sample_profile,
        title="AI Engineer Intern",
        company="Cognitive Systems",
        location="Remote",
        description="Looking for Python and PyTorch interns.",
    )

    assert result.success is True
    assert result.ai_score == 88.0
    assert result.recommendation == "STRONG_MATCH"
    assert "Python" in result.matched_requirements
    assert result.eligibility is True


# ---------------------------------------------------------------------------
# AIMatcher.evaluate — retry path
# ---------------------------------------------------------------------------

def test_ai_matcher_handles_retry_on_initial_invalid_json(sample_profile):
    """Verify AIMatcher retries with correction prompt if initial output is invalid JSON."""
    mock_client = MagicMock()
    # First response is invalid; second response succeeds
    mock_client.generate.side_effect = [
        "I cannot provide JSON right now.",
        '{"ai_score": 80, "recommendation": "STRONG_MATCH", "reason": "Fixed JSON", "matched_requirements": ["Python"], "missing_requirements": [], "risk_flags": [], "eligibility": true}',
    ]

    matcher = AIMatcher(client=mock_client)
    result = matcher.evaluate(
        profile=sample_profile,
        title="Python Intern",
        company="DevCorp",
        location="Remote",
        description="Python internship.",
    )

    assert mock_client.generate.call_count == 2
    assert result.success is True
    assert result.ai_score == 80.0


# ---------------------------------------------------------------------------
# AIMatcher.evaluate — graceful fallback (all attempts fail)
# ---------------------------------------------------------------------------

def test_ai_matcher_graceful_fallback_when_all_attempts_fail(sample_profile):
    """Verify AIMatcher falls back without crashing when all generation attempts fail.

    Critical: ai_score MUST be None (not 0.0) in the fallback result so that
    the hybrid scoring function knows AI truly failed.
    """
    mock_client = MagicMock()
    mock_client.generate.side_effect = RuntimeError("Ollama crashed")

    matcher = AIMatcher(client=mock_client)
    result = matcher.evaluate(
        profile=sample_profile,
        title="Python Intern",
        company="DevCorp",
        location="Remote",
        description="Python internship.",
    )

    assert result.success is False
    # Must be None — NOT 0.0 — so hybrid scoring correctly detects AI failure
    assert result.ai_score is None
    assert result.recommendation == "ANALYSIS_FAILED"
    assert "could not be completed" in result.reason


def test_ai_matcher_graceful_fallback_on_timeout(sample_profile):
    """Verify AIMatcher falls back correctly when Ollama times out."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = TimeoutError("Ollama timed out after 120s")

    matcher = AIMatcher(client=mock_client)
    result = matcher.evaluate(
        profile=sample_profile,
        title="ML Engineer",
        company="SlowCo",
        location="Remote",
        description="Machine learning role.",
    )

    assert result.success is False
    assert result.ai_score is None  # Must be None, not 0.0
    assert result.recommendation == "ANALYSIS_FAILED"


def test_ai_matcher_graceful_fallback_on_connection_error(sample_profile):
    """Verify AIMatcher falls back correctly when Ollama cannot be reached."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = ConnectionError("Could not connect to Ollama at http://localhost:11434")

    matcher = AIMatcher(client=mock_client)
    result = matcher.evaluate(
        profile=sample_profile,
        title="Data Scientist",
        company="RemoteCo",
        location="Remote",
        description="Data science role.",
    )

    assert result.success is False
    assert result.ai_score is None  # Must be None, not 0.0
    assert result.recommendation == "ANALYSIS_FAILED"


# ---------------------------------------------------------------------------
# calculate_hybrid_score
# ---------------------------------------------------------------------------

def test_calculate_hybrid_score_normal():
    """Verify the hybrid formula: 60% deterministic + 40% AI."""
    # 85 * 0.60 + 92 * 0.40 = 51.0 + 36.8 = 87.8
    final, category = calculate_hybrid_score(deterministic_score=85.0, ai_score=92.0, ai_success=True)
    assert final == 87.8
    assert category == MatchCategory.STRONG_MATCH


def test_calculate_hybrid_score_excellent():
    """Verify excellent match category from hybrid formula."""
    # 95 * 0.60 + 95 * 0.40 = 95.0
    final, category = calculate_hybrid_score(deterministic_score=95.0, ai_score=95.0, ai_success=True)
    assert final == 95.0
    assert category == MatchCategory.EXCELLENT_MATCH


def test_calculate_hybrid_score_fallback_when_ai_fails_none():
    """When ai_score is None (failure), deterministic score is used as final."""
    # Fallback: ai_score=None → use deterministic score unchanged
    final, category = calculate_hybrid_score(deterministic_score=84.0, ai_score=None, ai_success=False)
    assert final == 84.0
    assert category == MatchCategory.STRONG_MATCH


def test_calculate_hybrid_score_fallback_when_ai_success_false():
    """When ai_success=False regardless of ai_score value, deterministic score is preserved."""
    final, category = calculate_hybrid_score(deterministic_score=72.0, ai_score=50.0, ai_success=False)
    assert final == 72.0


def test_calculate_hybrid_score_zero_is_valid_ai_score():
    """A genuine ai_score of 0.0 with ai_success=True should produce a hybrid (not fallback) score."""
    # 60 * 0.60 + 0.0 * 0.40 = 36.0
    final, category = calculate_hybrid_score(deterministic_score=60.0, ai_score=0.0, ai_success=True)
    assert final == 36.0  # Hybrid formula applied — 0.0 is treated as a real (poor) score


def test_calculate_hybrid_score_none_with_success_true():
    """Even if ai_success=True but ai_score is None, fall back to deterministic (safety guard)."""
    # This should not happen in normal flow, but must not crash
    final, category = calculate_hybrid_score(deterministic_score=55.0, ai_score=None, ai_success=True)
    assert final == 55.0
