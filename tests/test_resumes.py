"""Tests for ResumeSelector and ResumeRecommendation."""

import pytest
from src.resumes.resume_selector import ResumeSelector, ResumeRecommendation


@pytest.fixture
def selector():
    return ResumeSelector()


def test_resume_recommendation_ai_ml(selector):
    rec = selector.recommend_resume(
        title="Machine Learning Engineer Intern",
        description="Looking for an intern experienced with Python, TensorFlow, and computer vision algorithms.",
        matched_skills=["Python", "TensorFlow", "Computer Vision"],
    )
    assert isinstance(rec, ResumeRecommendation)
    assert rec.category == "ai_ml"
    assert "artificial intelligence" in rec.category_name.lower() or "machine learning" in rec.category_name.lower()
    assert any("vision" in kw or "machine learning" in kw or "tensorflow" in kw for kw in rec.matching_keywords)


def test_resume_recommendation_embedded(selector):
    rec = selector.recommend_resume(
        title="Embedded Systems Engineer",
        description="Develop firmware and microcontroller drivers using C/C++ on ARM architecture.",
        matched_skills=["Electronics", "ECE"],
    )
    assert rec.category == "embedded"
    assert "embedded" in rec.category_name.lower()


def test_resume_recommendation_software(selector):
    rec = selector.recommend_resume(
        title="Python Backend Developer",
        description="Build scalable REST APIs using FastAPI and SQL databases. Solid DSA knowledge required.",
        matched_skills=["Python", "SQL", "FastAPI"],
    )
    assert rec.category == "software"
    assert "software" in rec.category_name.lower()


def test_resume_recommendation_general_fallback(selector):
    rec = selector.recommend_resume(
        title="Summer Analyst Trainee",
        description="General rotational program for new university graduates.",
    )
    assert rec.category == "general"


def test_list_available_resumes(selector):
    available = selector.list_available_resumes()
    assert "ai_ml" in available
    assert "embedded" in available
    assert "software" in available
    assert "general" in available
