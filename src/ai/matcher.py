"""
Evaluates job postings against a candidate profile across 6 key dimensions:
1. Role Match (25 pts)
2. Skills Match (30 pts)
3. Education Compatibility (15 pts)
4. Experience Level (15 pts)
5. Location Suitability (10 pts)
6. Employment Type (5 pts)
Total Maximum Score: 100 pts
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from src.models import UserProfile


class MatchCategory:
    """Standard match classification tiers."""
    EXCELLENT_MATCH = "EXCELLENT_MATCH"  # 90 - 100
    STRONG_MATCH = "STRONG_MATCH"        # 80 - 89
    GOOD_MATCH = "GOOD_MATCH"            # 70 - 79
    POSSIBLE_MATCH = "POSSIBLE_MATCH"    # 60 - 69
    WEAK_MATCH = "WEAK_MATCH"            # 40 - 59
    POOR_MATCH = "POOR_MATCH"            # 0 - 39


class DeterministicMatchResult(BaseModel):
    """Structured result of deterministic evaluation."""
    score: float = Field(..., description="Overall deterministic score (0 - 100)")
    role_score: float = Field(..., description="Role alignment score (0 - 25)")
    skills_score: float = Field(..., description="Skills alignment score (0 - 30)")
    education_score: float = Field(..., description="Education alignment score (0 - 15)")
    experience_score: float = Field(..., description="Experience level score (0 - 15)")
    location_score: float = Field(..., description="Location alignment score (0 - 10)")
    employment_score: float = Field(..., description="Employment type score (0 - 5)")
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both profile and job")
    missing_skills: List[str] = Field(default_factory=list, description="Profile skills missing or desired in job")
    match_category: str = Field(..., description="Categorical classification")
    recommendation: str = Field(..., description="Summary recommendation")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic breakdown")

    # New eligibility fields
    eligible: bool = Field(..., description="Whether the candidate is eligible for this role")
    eligibility_reason: str = Field(..., description="Reason for eligibility decision")


def categorize_score(score: float) -> str:
    """Map numeric score (0-100) to standard match category."""
    if score >= 90.0:
        return MatchCategory.EXCELLENT_MATCH
    if score >= 80.0:
        return MatchCategory.STRONG_MATCH
    if score >= 70.0:
        return MatchCategory.GOOD_MATCH
    if score >= 60.0:
        return MatchCategory.POSSIBLE_MATCH
    if score >= 40.0:
        return MatchCategory.WEAK_MATCH
    return MatchCategory.POOR_MATCH


# Common senior/leadership keywords that should be penalized for entry/intern profiles
SENIOR_KEYWORDS = [
    r"\bsenior\b", r"\bsr\.?\b", r"\blead\b", r"\bstaff\b", r"\bprincipal\b",
    r"\bdirector\b", r"\bhead\b", r"\bmanager\b", r"\bvp\b", r"\barchitect\b",
]

ENTRY_KEYWORDS = [
    r"\bintern\b", r"\binternship\b", r"\bgraduate\b", r"\bfresher\b",
    r"\bjunior\b", r"\bjr\.?\b", r"\bassociate\b", r"\btrainee\b", r"\bentry[- ]level\b",
]

NON_ENGINEERING_KEYWORDS = [
    r"\baccount executive\b", r"\bsales\b", r"\bmarketing\b", r"\brecruiter\b",
    r"\btalent acquisition\b", r"\baccount manager\b", r"\bbusiness development\b",
    r"\bcopywriter\b", r"\blegal counsel\b", r"\bhuman resources\b", r"\bhr\b",
    r"\bfinance\b", r"\bprocurement\b", r"\bcustomer success\b", r"\baccountant\b",
    r"\baccounting\b", r"\bpeople systems\b", r"\bpeople operations\b",
]

# Years-of-experience patterns.
_YEARS_TAIL = r"\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[a-z-]+){0,3}\s+experience\b"
HIGH_EXPERIENCE_PATTERN = re.compile(r"\b(?:[5-9]|[1-9][0-9])" + _YEARS_TAIL)  # 5+ years
MID_EXPERIENCE_PATTERN = re.compile(r"\b[3-4]" + _YEARS_TAIL)                  # 3-4 years
LOW_EXPERIENCE_PATTERN = re.compile(r"\b[1-2]" + _YEARS_TAIL)                  # 1-2 years

COMMON_TECH_SKILLS = [
    "python", "machine learning", "deep learning", "computer vision", "nlp",
    "pytorch", "tensorflow", "keras", "opencv", "scikit-learn", "pandas",
    "numpy", "fastapi", "flask", "django", "sql", "sqlite", "postgresql",
    "mysql", "git", "github", "docker", "kubernetes", "linux", "aws", "gcp",
    "azure", "react", "javascript", "typescript", "html", "css", "c++", "c",
    "java", "playwright", "selenium", "rest api", "graphql", "redis",
]


class DeterministicMatcher:
    """Evaluates candidate-job compatibility using deterministic scoring rules."""

    def __init__(self, profile: UserProfile):
        """Initialize matcher with candidate profile."""
        self.profile = profile
        self._normalized_skills = [s.strip().lower() for s in profile.skills]
        self._normalized_roles = [r.strip().lower() for r in profile.target_roles]
        self._normalized_locations = [loc.strip().lower() for loc in profile.preferred_locations]

    def evaluate(
        self,
        title: str,
        description: str,
        location: str,
        employment_type: Optional[str] = None,
    ) -> DeterministicMatchResult:
        """Score a job posting against candidate profile."""
        title_lower = (title or "").lower()
        desc_lower = (description or "").lower()
        loc_lower = (location or "").lower()
        emp_lower = (employment_type or "").lower()

        # 1. Role Match (25 pts)
        role_score, role_reason = self._score_role(title_lower)

        # 2. Skills Match (30 pts)
        skills_score, matched_skills, missing_skills = self._score_skills(title_lower, desc_lower)

        # 3. Education Match (15 pts)
        education_score, edu_reason = self._score_education(desc_lower)

        # 4. Experience Match (15 pts)
        experience_score, exp_reason = self._score_experience(title_lower, desc_lower)

        # 5. Location Match (10 pts)
        location_score, loc_reason = self._score_location(loc_lower)

        # 6. Employment Type Match (5 pts)
        employment_score, emp_reason = self._score_employment_type(title_lower, emp_lower)

        total_score = round(
            role_score + skills_score + education_score + experience_score + location_score + employment_score,
            1,
        )
        total_score = max(0.0, min(100.0, total_score))
        category = categorize_score(total_score)

        # ============================================================
        # NEW: Eligibility decision (hard rules for fresher/intern)
        # ============================================================
        eligible, eligibility_reason = self._check_eligibility(title_lower, desc_lower)

        return DeterministicMatchResult(
            score=total_score,
            role_score=role_score,
            skills_score=skills_score,
            education_score=education_score,
            experience_score=experience_score,
            location_score=location_score,
            employment_score=employment_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_category=category,
            recommendation=category.replace("_", " ").title(),
            details={
                "role_reason": role_reason,
                "education_reason": edu_reason,
                "experience_reason": exp_reason,
                "location_reason": loc_reason,
                "employment_reason": emp_reason,
            },
            eligible=eligible,
            eligibility_reason=eligibility_reason,
        )

    def _check_eligibility(self, title_lower: str, desc_lower: str) -> Tuple[bool, str]:
        """Hard eligibility rules for entry-level / fresher candidates."""

        # Rule 1: Non-engineering / sales / business roles → ineligible
        if any(re.search(pat, title_lower) for pat in NON_ENGINEERING_KEYWORDS):
            return False, "Non-engineering role (sales/marketing/HR/finance/customer success) – not suitable for technical candidate"

        # Rule 2: Senior / leadership titles → ineligible
        if any(re.search(pat, title_lower) for pat in SENIOR_KEYWORDS):
            return False, "Senior/leadership title detected – not suitable for fresher"

        # Rule 3: Explicit 5+ years experience requirement → ineligible
        if HIGH_EXPERIENCE_PATTERN.search(desc_lower):
            return False, "Requires 5+ years of professional experience"

        # Everything else is eligible (internships, entry-level, 0-2 years, etc.)
        return True, "Eligible – matches entry-level / fresher criteria"

    def _score_role(self, title_lower: str) -> Tuple[float, str]:
        """Score role compatibility against target roles (Max 25 pts)."""
        for target in self._normalized_roles:
            if target in title_lower or title_lower in target:
                return 25.0, f"Strong match for target role '{target}'"

        title_tokens = set(re.findall(r"\b[a-z]{2,}\b", title_lower))
        best_overlap_ratio = 0.0
        best_target = ""
        for target in self._normalized_roles:
            target_tokens = set(re.findall(r"\b[a-z]{2,}\b", target))
            if not target_tokens:
                continue
            common = title_tokens.intersection(target_tokens)
            ratio = len(common) / len(target_tokens)
            if ratio > best_overlap_ratio:
                best_overlap_ratio = ratio
                best_target = target

        if best_overlap_ratio >= 0.75:
            return 22.0, f"High title keyword overlap with '{best_target}'"
        if best_overlap_ratio >= 0.5:
            return 17.0, f"Moderate title keyword overlap with '{best_target}'"
        if best_overlap_ratio >= 0.3:
            return 10.0, f"Partial domain overlap with '{best_target}'"

        if any(w in title_lower for w in ["ai", "ml", "python", "data", "software", "developer", "engineer"]):
            return 8.0, "General domain keyword match"

        return 2.0, "Low role keyword match"

    def _score_skills(self, title_lower: str, desc_lower: str) -> Tuple[float, List[str], List[str]]:
        """Score skills overlap based on candidate skills and job requirements (Max 30 pts)."""
        combined_text = f"{title_lower} {desc_lower}"
        matched: List[str] = []
        missing: List[str] = []

        for skill in self.profile.skills:
            skill_pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(skill_pattern, combined_text):
                matched.append(skill)

        job_extra_skills: List[str] = []
        for common_skill in COMMON_TECH_SKILLS:
            if common_skill not in self._normalized_skills:
                skill_pat = r"\b" + re.escape(common_skill) + r"\b"
                if re.search(skill_pat, combined_text):
                    missing_label = common_skill.title()
                    if missing_label not in missing:
                        missing.append(missing_label)
                    job_extra_skills.append(common_skill)

        total_job_skills_count = len(matched) + len(job_extra_skills)
        if total_job_skills_count == 0:
            return 15.0, matched, missing

        coverage_ratio = len(matched) / total_job_skills_count
        points = (coverage_ratio * 20.0) + min(10.0, len(matched) * 3.5)
        points = round(max(0.0, min(30.0, points)), 1)
        return points, matched, missing[:10]

    def _score_education(self, desc_lower: str) -> Tuple[float, str]:
        """Score education compatibility (Max 15 pts)."""
        if re.search(r"\b(ph\.?d|doctorate)\b", desc_lower) and not re.search(r"\b(bachelor|master|b\.?tech)\b", desc_lower):
            return 4.0, "Requires advanced PhD degree"

        if any(re.search(pat, desc_lower) for pat in [
            r"\bcurrently pursuing\b", r"\benrolled in\b", r"\bundergraduate\b",
            r"\bbachelor'?s?\b", r"\bb\.?tech\b", r"\bcomputer science\b",
            r"\bclass of 202[4-7]\b", r"\bgraduating\b",
        ]):
            return 15.0, "Degree and major directly align with candidate background"

        return 12.0, "Standard or unstated educational requirements"

    def _score_experience(self, title_lower: str, desc_lower: str) -> Tuple[float, str]:
        """Score experience suitability, heavily penalizing senior roles (Max 15 pts)."""
        if any(re.search(pat, title_lower) for pat in SENIOR_KEYWORDS):
            return 2.0, "High senior/lead seniority title detected"

        if HIGH_EXPERIENCE_PATTERN.search(desc_lower):
            return 3.0, "Requires 5+ years professional experience"

        if MID_EXPERIENCE_PATTERN.search(desc_lower):
            return 7.0, "Requires 3-4 years experience"

        if any(re.search(pat, title_lower) for pat in ENTRY_KEYWORDS) or any(
            re.search(pat, desc_lower) for pat in [
                r"\b0[-–]1\s*years?\b", r"\b0[-–]2\s*years?\b", r"\bno experience required\b",
                r"\binternship\b", r"\bfreshers?\b",
            ]
        ):
            return 15.0, "Targeted for interns and entry-level applicants"

        if LOW_EXPERIENCE_PATTERN.search(desc_lower):
            return 11.0, "Requires 1-2 years experience (accessible for fresh graduates)"

        return 10.0, "Experience requirement not explicitly elevated"

    def _score_location(self, loc_lower: str) -> Tuple[float, str]:
        """Score location alignment against preferences (Max 10 pts)."""
        if any(w in loc_lower for w in ["remote", "anywhere", "work from home", "virtual"]):
            if self.profile.work_preferences.remote:
                return 10.0, "Fully remote opportunity"

        for pref in self._normalized_locations:
            if pref in loc_lower:
                return 10.0, f"Located in preferred location '{pref.title()}'"

        if "india" in loc_lower:
            return 8.0, "Nationwide opportunity in India"

        if not loc_lower or "unspecified" in loc_lower:
            return 6.0, "Location unspecified"

        return 2.0, "Location outside primary preferences"

    def _score_employment_type(self, title_lower: str, emp_lower: str) -> Tuple[float, str]:
        """Score employment type (Max 5 pts)."""
        is_intern = "intern" in title_lower or "intern" in emp_lower
        is_fulltime = "full" in emp_lower or ("intern" not in title_lower and "contract" not in emp_lower)

        for target_type in self.profile.employment_types:
            target_lower = target_type.lower()
            if "intern" in target_lower and is_intern:
                return 5.0, "Direct match for Internship"
            if "full" in target_lower and is_fulltime:
                return 5.0, "Direct match for Full-time"

        return 2.0, "Partial employment type match"