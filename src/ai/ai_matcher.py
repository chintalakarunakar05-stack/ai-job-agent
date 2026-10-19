"""Semantic AI Matcher using local LLM inference.

Evaluates high-potential candidate jobs using Ollama for deep semantic reasoning,
validating responses against strict Pydantic schemas with automatic JSON extraction and retry.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

from src.ai.matcher import MatchCategory, categorize_score
from src.ai.ollama_client import OllamaClient
from src.models import UserProfile

logger = logging.getLogger(__name__)


class AIMatchResult(BaseModel):
    """Validated structured output of AI semantic matching.

    When AI inference fails, ai_score is set to None (not 0.0) so that
    the hybrid scoring function can distinguish a real zero score from a
    failure. The `success` flag provides an additional explicit signal.
    """
    ai_score: Optional[float] = Field(
        default=None,
        description="Semantic match score (0 - 100). None means AI inference did not produce a result.",
    )
    recommendation: str = Field(default="ANALYSIS_FAILED", description="Categorical recommendation")
    reason: str = Field(default="", description="Detailed explanation of fit or mismatch")
    matched_requirements: List[str] = Field(default_factory=list, description="Requirements candidate satisfies")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements candidate may lack")
    risk_flags: List[str] = Field(default_factory=list, description="Potential blockers or mismatches")
    eligibility: bool = Field(default=True, description="Whether candidate meets minimum criteria")
    success: bool = Field(default=True, description="Whether AI inference succeeded")

    @field_validator("ai_score")
    @classmethod
    def clamp_score(cls, v: Optional[float]) -> Optional[float]:
        """Ensure score is within 0 to 100 range; pass through None unchanged."""
        if v is None:
            return None
        return max(0.0, min(100.0, float(v)))


SYSTEM_PROMPT = """You are an expert AI Career Matchmaker and Technical Recruiter.
Evaluate the compatibility between the candidate and the job posting.

Your response MUST be a single raw JSON object with this exact schema:
{
    "ai_score": <number between 0 and 100>,
    "recommendation": "<EXCELLENT_MATCH | STRONG_MATCH | GOOD_MATCH | POSSIBLE_MATCH | WEAK_MATCH | POOR_MATCH>",
    "reason": "<2-3 sentence clear explanation>",
    "matched_requirements": ["<key skill/requirement matched>", ...],
    "missing_requirements": ["<missing requirement or skill>", ...],
    "risk_flags": ["<risk or red flag, e.g. requires 5+ years experience, requires relocation outside country>"],
    "eligibility": <true or false>
}

Be realistic and objective. Do not hallucinate qualifications the candidate does not have.
Return ONLY valid JSON. Do not include markdown formatting or extra commentary outside the JSON."""


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Safely extract and parse JSON object from text or markdown blocks."""
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Extract content between ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Find outermost curly braces { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate_json = text[start : end + 1]
        try:
            return json.loads(candidate_json)
        except Exception:
            pass

    return None


class AIMatcher:
    """Performs LLM-based semantic analysis of job-candidate compatibility."""

    def __init__(self, client: Optional[OllamaClient] = None):
        """Initialize with an OllamaClient instance."""
        self.client = client or OllamaClient()

    def evaluate(
        self,
        profile: UserProfile,
        title: str,
        company: str,
        location: str,
        description: str,
        employment_type: Optional[str] = None,
    ) -> AIMatchResult:
        """Perform semantic analysis via LLM with retry and fallback.

        Args:
            profile: Candidate UserProfile.
            title: Job title.
            company: Company name.
            location: Job location.
            description: Cleaned job description text.
            employment_type: Optional employment type string.

        Returns:
            Validated AIMatchResult. If LLM fails, returns fallback result with success=False.
        """
        # Truncate overly verbose descriptions to keep prompt compact and fast
        truncated_desc = description[:1500] if len(description) > 1500 else description

        prompt = self._build_prompt(
            profile=profile,
            title=title,
            company=company,
            location=location,
            description=truncated_desc,
            employment_type=employment_type,
        )

        # Attempt 1: Standard generation
        try:
            raw_response = self.client.generate(prompt=prompt, system=SYSTEM_PROMPT, json_mode=True)
            parsed_dict = extract_json_from_text(raw_response)
            if parsed_dict:
                return AIMatchResult.model_validate(parsed_dict)
            logger.warning(f"LLM response was not valid JSON. Content was: {repr(raw_response[:300])}. Retrying...")
        except Exception as exc:
            logger.warning(f"Initial AI evaluation call failed: {exc}. Attempting retry...")

        # Attempt 2: Correction prompt retry
        try:
            correction_prompt = (
                f"{prompt}\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Respond ONLY with a valid, parsable JSON object conforming to the required schema."
            )
            raw_retry = self.client.generate(prompt=correction_prompt, system=SYSTEM_PROMPT, json_mode=True)
            parsed_retry = extract_json_from_text(raw_retry)
            if parsed_retry:
                return AIMatchResult.model_validate(parsed_retry)
        except Exception as exc:
            logger.error(f"Correction retry failed: {exc}")

        # Graceful fallback: return failure object without crashing the pipeline.
        # ai_score=None signals that AI did not produce a result — it is NOT a score of zero.
        # The hybrid scoring function checks for None explicitly.
        logger.error(f"AI evaluation could not be completed for '{company} - {title}'. Using fallback.")
        return AIMatchResult(
            ai_score=None,
            recommendation="ANALYSIS_FAILED",
            reason="AI semantic inference could not be completed. Deterministic score preserved.",
            matched_requirements=[],
            missing_requirements=[],
            risk_flags=["AI analysis timed out or returned malformed output"],
            eligibility=True,
            success=False,
        )

    def _build_prompt(
        self,
        profile: UserProfile,
        title: str,
        company: str,
        location: str,
        description: str,
        employment_type: Optional[str],
    ) -> str:
        """Format candidate and job details into a concise analysis prompt."""
        edu_summary = [
            f"{e.degree} in {e.field_of_study} from {e.institution} (Graduation: {e.graduation_year})"
            for e in profile.education
        ]
        proj_summary = [f"{p.title}: {p.description}" for p in profile.projects[:3]]

        return f"""### CANDIDATE PROFILE
Name: {profile.name}
Experience Level: {profile.experience_level}
Target Roles: {', '.join(profile.target_roles)}
Skills: {', '.join(profile.skills)}
Education: {'; '.join(edu_summary) if edu_summary else 'Computer Science / Engineering'}
Key Projects: {'; '.join(proj_summary) if proj_summary else 'Academic engineering projects'}
Preferred Locations: {', '.join(profile.preferred_locations)}
Preferred Employment Types: {', '.join(profile.employment_types)}

### JOB OPPORTUNITY
Company: {company}
Title: {title}
Location: {location}
Employment Type: {employment_type or 'Not specified'}

Description:
{description}

Provide your structured JSON evaluation now."""


def calculate_hybrid_score(
    deterministic_score: float,
    ai_score: Optional[float],
    ai_success: bool = True,
    weight_deterministic: float = 0.60,
    weight_ai: float = 0.40,
) -> Tuple[float, str]:
    """Calculate combined hybrid score and category.

    Hybrid formula (when AI succeeds):
        final = deterministic_score * 0.60 + ai_score * 0.40

    Fallback (when AI fails or is skipped):
        final = deterministic_score
        This preserves the deterministic result without fabricating an AI contribution.

    Args:
        deterministic_score: Score from rule-based matcher (0 - 100).
        ai_score: Score from LLM semantic matcher (0 - 100).
                  Must be None when AI failed — do NOT pass 0.0 to signal failure.
        ai_success: Whether the AI evaluation succeeded (additional explicit guard).
        weight_deterministic: Weight for deterministic score (default 0.60).
        weight_ai: Weight for AI score (default 0.40).

    Returns:
        Tuple of (final_score: float, match_category: str).
    """
    # Use hybrid only when AI produced a real result (ai_score is not None AND success=True).
    # ai_score=0.0 is treated as a valid (but poor) AI score — NOT a failure signal.
    if not ai_success or ai_score is None:
        # AI was skipped, failed, or timed out; rely entirely on deterministic score.
        final = round(deterministic_score, 1)
    else:
        final = round((deterministic_score * weight_deterministic) + (ai_score * weight_ai), 1)

    final = max(0.0, min(100.0, final))
    category = categorize_score(final)
    return final, category
