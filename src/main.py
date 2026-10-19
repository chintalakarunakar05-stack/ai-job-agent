"""Main entry point for AI Job & Internship Application Agent.

Initializes logging, bootstraps the database, verifies system health,
and provides rich FastAPI endpoints for job discovery, matching, and application workflow.
"""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for both module and direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.ai.matching_manager import MatchingManager
from src.collectors.manager import CollectorManager
from src.config import settings
from src.database import get_db, init_db
from src.models import (
    ApplicationModel,
    ApplicationStatus,
    JobModel,
    UserProfile,
    load_user_profile,
)
from src.resumes.resume_selector import ResumeSelector

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOGS_DIR / "agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ai_job_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Starting AI Job & Internship Application Agent...")
    init_db()
    try:
        profile = load_user_profile()
        logger.info(f"Loaded candidate profile for: {profile.name} ({profile.email})")
    except Exception as exc:
        logger.warning(f"Profile could not be loaded at startup: {exc}")
    yield
    logger.info("Shutting down AI Job & Internship Application Agent...")


# FastAPI application instance
app = FastAPI(
    title="AI Job & Internship Application Agent API",
    description="Backend API powering job discovery, resume matching, and human-in-the-loop application workflow.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC REQUEST / RESPONSE SCHEMAS
# ============================================================================

class StatusUpdateSchema(BaseModel):
    status: ApplicationStatus
    notes: Optional[str] = None


class CollectRequestSchema(BaseModel):
    dry_run: bool = False


class MatchRequestSchema(BaseModel):
    dry_run: bool = False
    no_ai: bool = False
    limit: Optional[int] = None


class ApplicationCreateSchema(BaseModel):
    job_id: int
    resume_used: Optional[str] = None
    cover_letter: Optional[str] = None
    notes: Optional[str] = None


class JobDetailResponse(BaseModel):
    id: int
    title: str
    company: str
    location: str
    description: str
    url: str
    source: str
    employment_type: Optional[str] = None
    salary: Optional[str] = None
    status: ApplicationStatus
    match_score: Optional[float] = None
    deterministic_score: Optional[float] = None
    ai_score: Optional[float] = None
    final_score: Optional[float] = None
    match_category: Optional[str] = None
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    match_reason: Optional[str] = None
    ai_analysis: Optional[Dict[str, Any]] = None
    recommended_resume: Optional[str] = None
    collected_at: str

    model_config = {"from_attributes": True}


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/health", tags=["Health"])
def health_check():
    """Verify application health and configuration status."""
    return {
        "status": "healthy",
        "app_env": settings.APP_ENV,
        "ai_provider": settings.AI_PROVIDER,
        "ollama_model": settings.OLLAMA_MODEL,
        "database_url": settings.DATABASE_URL.split("///")[-1],
    }


@app.get("/api/profile", tags=["Profile"], response_model=UserProfile)
def get_profile():
    """Retrieve the candidate's active profile."""
    try:
        return load_user_profile()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {exc}")


@app.put("/api/profile", tags=["Profile"], response_model=UserProfile)
def update_profile(updated_profile: UserProfile):
    """Save updated profile to profile.json."""
    try:
        with open(settings.PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_profile.model_dump(), f, indent=2)
        logger.info(f"Updated profile for: {updated_profile.name}")
        return updated_profile
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {exc}")


@app.get("/api/stats", tags=["Dashboard"])
def get_stats(db: Session = Depends(get_db)):
    """Retrieve aggregate statistics for dashboard rendering."""
    total_jobs = db.query(JobModel).count()
    status_counts = {}
    for st in ApplicationStatus:
        cnt = db.query(JobModel).filter(JobModel.status == st).count()
        if cnt > 0:
            status_counts[st.value] = cnt

    high_matches = db.query(JobModel).filter(JobModel.match_score >= 80.0).count()
    medium_matches = db.query(JobModel).filter(
        JobModel.match_score >= 60.0, JobModel.match_score < 80.0
    ).count()

    total_applications = db.query(ApplicationModel).count()

    return {
        "total_jobs": total_jobs,
        "high_matches": high_matches,
        "medium_matches": medium_matches,
        "status_breakdown": status_counts,
        "total_applications": total_applications,
        "ai_model": settings.OLLAMA_MODEL,
        "ai_provider": settings.AI_PROVIDER,
    }


@app.get("/api/jobs", tags=["Jobs"])
def list_jobs(
    status: Optional[str] = Query(None, description="Filter by ApplicationStatus"),
    min_score: Optional[float] = Query(None, description="Minimum match score"),
    search: Optional[str] = Query(None, description="Search keyword in title or company"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List discovered jobs with optional filters, ordering by match score descending."""
    query = db.query(JobModel)

    if status:
        query = query.filter(JobModel.status == status)

    if min_score is not None:
        query = query.filter(JobModel.match_score >= min_score)

    if search:
        search_fmt = f"%{search}%"
        query = query.filter(
            (JobModel.title.ilike(search_fmt)) | (JobModel.company.ilike(search_fmt))
        )

    total = query.count()
    jobs = query.order_by(desc(JobModel.match_score), desc(JobModel.collected_at)).offset(offset).limit(limit).all()

    def serialize_job(j: JobModel) -> Dict[str, Any]:
        matched = []
        missing = []
        ai_meta = None
        if j.matched_skills:
            try:
                matched = json.loads(j.matched_skills)
            except Exception:
                pass
        if j.missing_skills:
            try:
                missing = json.loads(j.missing_skills)
            except Exception:
                pass
        if j.ai_analysis:
            try:
                ai_meta = json.loads(j.ai_analysis)
            except Exception:
                pass

        return {
            "id": j.id,
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "description": j.description[:300] + "..." if len(j.description) > 300 else j.description,
            "url": j.url,
            "source": j.source,
            "employment_type": j.employment_type,
            "salary": j.salary,
            "status": j.status.value,
            "match_score": j.match_score,
            "deterministic_score": j.deterministic_score,
            "ai_score": j.ai_score,
            "final_score": j.final_score,
            "match_category": j.match_category,
            "matched_skills": matched,
            "missing_skills": missing,
            "match_reason": j.match_reason,
            "ai_analysis": ai_meta,
            "recommended_resume": j.recommended_resume,
            "collected_at": j.collected_at.isoformat() if j.collected_at else "",
        }

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "jobs": [serialize_job(j) for j in jobs],
    }


@app.get("/api/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: int, db: Session = Depends(get_db)):
    """Retrieve complete details for a specific job."""
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    matched = []
    missing = []
    ai_meta = None
    if job.matched_skills:
        try:
            matched = json.loads(job.matched_skills)
        except Exception:
            pass
    if job.missing_skills:
        try:
            missing = json.loads(job.missing_skills)
        except Exception:
            pass
    if job.ai_analysis:
        try:
            ai_meta = json.loads(job.ai_analysis)
        except Exception:
            pass

    return {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "description": job.description,
        "url": job.url,
        "source": job.source,
        "employment_type": job.employment_type,
        "salary": job.salary,
        "status": job.status.value,
        "match_score": job.match_score,
        "deterministic_score": job.deterministic_score,
        "ai_score": job.ai_score,
        "final_score": job.final_score,
        "match_category": job.match_category,
        "matched_skills": matched,
        "missing_skills": missing,
        "match_reason": job.match_reason,
        "ai_analysis": ai_meta,
        "recommended_resume": job.recommended_resume,
        "collected_at": job.collected_at.isoformat() if job.collected_at else "",
    }


@app.patch("/api/jobs/{job_id}/status", tags=["Jobs"])
def update_job_status(job_id: int, req: StatusUpdateSchema, db: Session = Depends(get_db)):
    """Update application lifecycle status of a job."""
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.status = req.status
    db.commit()
    return {"id": job.id, "status": job.status.value, "updated": True}


@app.get("/api/resumes/recommend/{job_id}", tags=["Resumes"])
def recommend_resume_for_job(job_id: int, db: Session = Depends(get_db)):
    """Recommend the optimal resume track and tailoring advice for a job."""
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    matched_skills = []
    if job.matched_skills:
        try:
            matched_skills = json.loads(job.matched_skills)
        except Exception:
            pass

    selector = ResumeSelector()
    rec = selector.recommend_resume(
        title=job.title,
        description=job.description,
        matched_skills=matched_skills,
    )
    return rec


@app.post("/api/collect", tags=["Automation"])
def trigger_collection(req: CollectRequestSchema):
    """Trigger a live ATS job collection cycle."""
    manager = CollectorManager()
    stats = manager.collect_all(dry_run=req.dry_run)
    return {"message": "Job collection complete", "stats": stats}


@app.post("/api/match", tags=["Automation"])
def trigger_matching(req: MatchRequestSchema):
    """Trigger a hybrid deterministic + AI matching cycle."""
    manager = MatchingManager()
    stats = manager.run_matching(ai_limit=req.limit, dry_run=req.dry_run, no_ai=req.no_ai)
    return {"message": "Matching cycle complete", "stats": stats}


@app.get("/api/applications", tags=["Applications"])
def list_applications(db: Session = Depends(get_db)):
    """List all tracked applications."""
    apps = db.query(ApplicationModel).order_by(desc(ApplicationModel.created_at)).all()
    results = []
    for a in apps:
        results.append({
            "id": a.id,
            "job_id": a.job_id,
            "company": a.company,
            "role": a.role,
            "resume_used": a.resume_used,
            "status": a.status.value,
            "applied_date": a.applied_date.isoformat() if a.applied_date else None,
            "notes": a.notes,
            "created_at": a.created_at.isoformat() if a.created_at else "",
        })
    return results


@app.post("/api/applications", tags=["Applications"], status_code=status.HTTP_201_CREATED)
def create_application(req: ApplicationCreateSchema, db: Session = Depends(get_db)):
    """Create a new tracked application record."""
    job = db.query(JobModel).filter(JobModel.id == req.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    new_app = ApplicationModel(
        job_id=job.id,
        company=job.company,
        role=job.title,
        resume_used=req.resume_used,
        cover_letter=req.cover_letter,
        notes=req.notes,
        status=ApplicationStatus.REVIEW_REQUIRED,
    )
    db.add(new_app)
    job.status = ApplicationStatus.REVIEW_REQUIRED
    db.commit()
    db.refresh(new_app)
    return {"id": new_app.id, "job_id": new_app.job_id, "status": new_app.status.value}


def main():
    """CLI launcher entry point."""
    logger.info("Initializing database and verifying profile...")
    init_db()
    profile = load_user_profile()
    print("=" * 60)
    print(f"Candidate Profile: {profile.name}")
    print(f"Target Roles: {', '.join(profile.target_roles)}")
    print(f"Skills: {len(profile.skills)} technical skills loaded")
    print(f"AI Provider: {settings.AI_PROVIDER} ({settings.OLLAMA_MODEL})")
    print(f"Database: {settings.DATABASE_URL}")
    print("=" * 60)
    print("AI Job Agent is initialized and ready.")
    print("Run `uvicorn src.main:app --reload` to start the backend API server.")
    print("Run `streamlit run app/dashboard.py` to start the dashboard (Phase 7).")


if __name__ == "__main__":
    main()
