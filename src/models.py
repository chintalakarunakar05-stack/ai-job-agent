"""Data models and schemas for AI Job & Internship Application Agent.

Includes:
- Pydantic models for User Profile validation
- Pydantic models for normalized Job and Application data transfers
- SQLAlchemy ORM models for database persistence
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class ApplicationStatus(str, Enum):
    """Lifecycle status of a job opportunity / application."""
    NEW = "NEW"
    MATCHED = "MATCHED"
    SHORTLISTED = "SHORTLISTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    READY_TO_APPLY = "READY_TO_APPLY"
    APPLIED = "APPLIED"
    INTERVIEW = "INTERVIEW"
    ASSESSMENT = "ASSESSMENT"
    REJECTED = "REJECTED"
    OFFER = "OFFER"
    WITHDRAWN = "WITHDRAWN"
    FAILED = "FAILED"


# ============================================================================
# PYDANTIC SCHEMAS - USER PROFILE
# ============================================================================

class EducationItem(BaseModel):
    """Education record in user profile."""
    degree: str = Field(..., description="e.g. Bachelor of Technology")
    field_of_study: str = Field(..., description="e.g. Computer Science, Artificial Intelligence")
    institution: str = Field(..., description="College or University name")
    graduation_year: int = Field(..., description="Graduation year e.g. 2026")
    gpa: Optional[str] = Field(default=None, description="e.g. 8.5/10 or 3.8/4.0")


class ProjectItem(BaseModel):
    """Portfolio or academic project in user profile."""
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Summary of problem solved and impact")
    technologies: List[str] = Field(default_factory=list, description="List of libraries/tools used")
    link: Optional[str] = Field(default=None, description="GitHub repository or live URL")


class CertificationItem(BaseModel):
    """Professional or technical certification."""
    name: str = Field(..., description="Certification name")
    issuer: str = Field(..., description="Issuing organization")
    year: Optional[int] = Field(default=None, description="Year obtained")


class WorkPreference(BaseModel):
    """Working style and arrangement preferences."""
    remote: bool = Field(default=True, description="Open to fully remote roles")
    hybrid: bool = Field(default=True, description="Open to hybrid arrangements")
    on_site: bool = Field(default=True, description="Open to on-site roles")
    willing_to_relocate: bool = Field(default=False, description="Willingness to relocate")


class SalaryPreference(BaseModel):
    """Salary expectations."""
    currency: str = Field(default="INR", description="e.g. INR, USD, EUR")
    minimum: Optional[int] = Field(default=None, description="Minimum acceptable monthly/annual salary")
    target: Optional[int] = Field(default=None, description="Target desirable salary")


class UserProfile(BaseModel):
    """Complete candidate profile schema."""
    name: str = Field(..., description="Full legal name")
    email: str = Field(..., description="Contact email address")
    phone: str = Field(..., description="Phone number with country code")
    location: str = Field(..., description="Current city, country e.g. Hyderabad, India")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github: Optional[str] = Field(default=None, description="GitHub profile URL")
    portfolio: Optional[str] = Field(default=None, description="Portfolio website URL")
    
    summary: str = Field(default="", description="Professional bio or executive summary")
    education: List[EducationItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list, description="Primary technical and domain skills")
    target_roles: List[str] = Field(default_factory=list, description="Roles of interest e.g. 'AI Engineer Intern'")
    preferred_locations: List[str] = Field(default_factory=list, description="Preferred work cities or 'Remote'")
    employment_types: List[str] = Field(
        default_factory=lambda: ["Internship", "Full-time"],
        description="e.g. Internship, Full-time, Contract"
    )
    experience_level: str = Field(default="Entry Level / Intern", description="Current career level")
    salary_preferences: SalaryPreference = Field(default_factory=SalaryPreference)
    work_preferences: WorkPreference = Field(default_factory=WorkPreference)
    certifications: List[CertificationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=lambda: ["English"], description="Spoken/written languages")


def load_user_profile(path: Optional[Any] = None) -> UserProfile:
    """Load and validate candidate profile from JSON file.

    Args:
        path: Optional Path or str pointing to profile.json. Defaults to settings.PROFILE_PATH.

    Returns:
        Validated UserProfile instance.

    Raises:
        FileNotFoundError: If profile file does not exist.
        ValueError: If JSON is invalid or fails schema validation.
    """
    import json
    from pathlib import Path
    from src.config import settings

    target_path = Path(path) if path else settings.PROFILE_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"User profile file not found at: '{target_path.resolve()}'. "
            "Please create your candidate profile based on data/profile.json."
        )

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return UserProfile.model_validate(raw_data)
    except json.JSONDecodeError as err:
        raise ValueError(f"Profile JSON at '{target_path}' contains malformed JSON: {err}") from err
    except Exception as err:
        raise ValueError(f"Profile validation failed for '{target_path}': {err}") from err


# ============================================================================
# PYDANTIC SCHEMAS - JOB & APPLICATION TRANSFER OBJECTS
# ============================================================================

class JobBase(BaseModel):
    """Normalized job schema for collectors and processing."""
    title: str
    company: str
    location: str
    description: str
    url: str
    source: str
    source_job_id: Optional[str] = None
    employment_type: Optional[str] = "Full-time"
    salary: Optional[str] = None
    posted_date: Optional[datetime] = None
    application_url: Optional[str] = None


class JobCreate(JobBase):
    pass


class JobRead(JobBase):
    id: int
    match_score: Optional[float] = None
    match_reason: Optional[str] = None
    missing_skills: Optional[str] = None
    recommended_resume: Optional[str] = None
    status: ApplicationStatus
    collected_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# SQLALCHEMY ORM MODELS
# ============================================================================

class JobModel(Base):
    """Database entity representing a discovered job opportunity."""
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    source_job_id = Column(String(128), nullable=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    location = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    url = Column(String(1024), nullable=False, unique=True, index=True)
    source = Column(String(64), nullable=False)  # greenhouse, lever, custom, etc.
    employment_type = Column(String(64), nullable=True, default="Full-time")
    salary = Column(String(128), nullable=True)
    posted_date = Column(DateTime, nullable=True)
    application_url = Column(String(1024), nullable=True)

    # Analysis & Matching Results
    match_score = Column(Float, nullable=True, default=0.0)
    deterministic_score = Column(Float, nullable=True, default=0.0)
    ai_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True, default=0.0)
    match_category = Column(String(64), nullable=True)
    matched_skills = Column(Text, nullable=True)
    missing_skills = Column(Text, nullable=True)
    match_reason = Column(Text, nullable=True)
    ai_analysis = Column(Text, nullable=True)
    analyzed_at = Column(DateTime, nullable=True)
    recommended_resume = Column(String(255), nullable=True)

    # Application Status
    status = Column(SQLEnum(ApplicationStatus), default=ApplicationStatus.NEW, nullable=False)

    # Timestamps
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    applications = relationship("ApplicationModel", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<JobModel(id={self.id}, title='{self.title}', company='{self.company}', status='{self.status}')>"


class ApplicationModel(Base):
    """Database entity representing an application attempt/submission for a job."""
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    company = Column(String(255), nullable=False)
    role = Column(String(255), nullable=False)
    resume_used = Column(String(255), nullable=True)
    cover_letter = Column(Text, nullable=True)
    generated_answers = Column(Text, nullable=True)  # JSON serialized dictionary of Q&A
    form_data = Column(Text, nullable=True)  # JSON serialized fields filled

    status = Column(SQLEnum(ApplicationStatus), default=ApplicationStatus.REVIEW_REQUIRED, nullable=False)
    human_approved = Column(Boolean, default=False, nullable=False)
    applied_date = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    screenshot_path = Column(String(512), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    job = relationship("JobModel", back_populates="applications")

    def __repr__(self) -> str:
        return f"<ApplicationModel(id={self.id}, company='{self.company}', role='{self.role}', status='{self.status}')>"
