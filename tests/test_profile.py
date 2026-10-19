"""Unit tests for user profile validation and schema integrity."""

import pytest
from src.models import UserProfile, load_user_profile


def test_load_valid_user_profile():
    """Verify loading the default profile.json passes schema validation."""
    profile = load_user_profile()

    assert isinstance(profile, UserProfile)
    assert profile.name != ""
    assert len(profile.skills) > 0
    assert "Python" in profile.skills
    assert len(profile.target_roles) > 0
    assert len(profile.education) > 0
    assert profile.education[0].graduation_year == 2027


def test_profile_validation_rejects_missing_required_fields():
    """Verify validation raises an error if mandatory fields are missing."""
    with pytest.raises(Exception):
        UserProfile(
            email="invalid@example.com"
            # Missing required fields, including name, phone, and location.
        )


def test_profile_work_preferences():
    """Verify work preferences default correctly."""
    profile = load_user_profile()

    assert profile.work_preferences.remote is True
    assert profile.salary_preferences.currency == "INR"
