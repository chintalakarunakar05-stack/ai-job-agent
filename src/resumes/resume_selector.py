"""Resume Selector and Category Recommender.

Matches job descriptions against candidate resume categories and identifies
relevant resume tracks and keyword alignments.
"""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.config import settings


class ResumeRecommendation(BaseModel):
    """Recommended resume category and matching rationale."""
    category: str = Field(..., description="Resume track e.g. ai_ml, embedded, software, general")
    category_name: str = Field(..., description="Human-friendly category name")
    suggested_filename: Optional[str] = Field(default=None, description="Filename if present on disk")
    matching_keywords: List[str] = Field(default_factory=list, description="Keywords justifying this track")
    summary: str = Field(..., description="Summary advice for the candidate")


# Track category keyword definitions
CATEGORY_KEYWORDS: Dict[str, Dict[str, any]] = {
    "ai_ml": {
        "name": "Artificial Intelligence & Machine Learning Track",
        "keywords": [
            "ai", "artificial intelligence", "machine learning", "ml", "deep learning",
            "computer vision", "opencv", "tensorflow", "keras", "pytorch",
            "nlp", "natural language", "llm", "neural network", "data science"
        ],
        "default_file": "Karunakar_AI_ML_Resume.pdf",
    },
    "embedded": {
        "name": "Embedded Systems & Electronics Track",
        "keywords": [
            "embedded", "firmware", "microcontroller", "iot", "rtos",
            "arm", "arduino", "raspberry pi", "electronics", "ece",
            "pcb", "vlsi", "fpga", "sensors", "hardware", "c/c++"
        ],
        "default_file": "Karunakar_Embedded_Resume.pdf",
    },
    "software": {
        "name": "Software Engineering & Python Track",
        "keywords": [
            "software engineer", "software developer", "python", "backend",
            "fastapi", "django", "flask", "api", "rest", "full stack",
            "database", "sql", "git", "docker", "algorithms", "dsa", "oop"
        ],
        "default_file": "Karunakar_Software_Resume.pdf",
    },
    "general": {
        "name": "General Technical & Graduate Trainee Track",
        "keywords": ["graduate", "intern", "trainee", "associate", "entry level", "fresher"],
        "default_file": "Karunakar_General_Resume.pdf",
    },
}


class ResumeSelector:
    """Selects and recommends optimal resume variations for given job postings."""

    def __init__(self, resumes_dir: Optional[Path] = None):
        """Initialize selector with base resumes directory."""
        self.resumes_dir = resumes_dir or settings.RESUMES_DIR

    def list_available_resumes(self) -> Dict[str, List[str]]:
        """Scan resumes directory and list physical files under each track.

        Returns:
            Dictionary mapping category keys to list of found filenames.
        """
        available: Dict[str, List[str]] = {}
        for cat in CATEGORY_KEYWORDS:
            cat_dir = self.resumes_dir / cat
            if cat_dir.exists() and cat_dir.is_dir():
                files = [f.name for f in cat_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
                available[cat] = sorted(files)
            else:
                available[cat] = []
        return available

    def recommend_resume(
        self,
        title: str,
        description: str,
        matched_skills: Optional[List[str]] = None,
    ) -> ResumeRecommendation:
        """Analyze job content and recommend the most fitting resume track.

        Args:
            title: Job title.
            description: Job description text.
            matched_skills: Optional list of identified candidate skills.

        Returns:
            ResumeRecommendation with selected track and guidance.
        """
        text = f"{title} {description} {' '.join(matched_skills or [])}".lower()

        scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS}
        keyword_hits: Dict[str, List[str]] = {cat: [] for cat in CATEGORY_KEYWORDS}

        for cat, meta in CATEGORY_KEYWORDS.items():
            for kw in meta["keywords"]:
                if kw in text:
                    # Title matches carry 3x weight
                    weight = 3 if kw in title.lower() else 1
                    scores[cat] += weight
                    keyword_hits[cat].append(kw)

        # Prioritize ai_ml and embedded if tied with general/software
        best_cat = max(
            ["ai_ml", "embedded", "software", "general"],
            key=lambda c: scores[c]
        )

        # If zero keywords matched, default to general
        if scores[best_cat] == 0:
            best_cat = "general"

        meta = CATEGORY_KEYWORDS[best_cat]
        hits = list(dict.fromkeys(keyword_hits[best_cat]))[:8]  # unique top 8

        # Check for actual physical file in resumes directory
        available = self.list_available_resumes()
        chosen_file = available.get(best_cat, [None])[0] if available.get(best_cat) else meta["default_file"]

        summary = (
            f"Tailor profile around {meta['name']} highlighting "
            f"relevant projects ({', '.join(hits[:4]) if hits else 'core skills'})."
        )

        return ResumeRecommendation(
            category=best_cat,
            category_name=meta["name"],
            suggested_filename=chosen_file,
            matching_keywords=hits,
            summary=summary,
        )
