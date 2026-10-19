"""Interactive Streamlit Dashboard for AI Job & Internship Application Agent.

Allows candidates to:
- Monitor ATS ingestion and AI matching pipelines.
- Browse, search, filter, and inspect job opportunities with detailed match scores.
- Review matched skills, missing skills, and LLM semantic recommendations.
- Select optimal resume tracks.
- Move jobs through the human-in-the-loop application lifecycle.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.matching_manager import MatchingManager
from src.collectors.manager import CollectorManager
from src.config import settings
from src.database import get_db_session, init_db
from src.models import (
    ApplicationModel,
    ApplicationStatus,
    JobModel,
    UserProfile,
    load_user_profile,
)
from src.resumes.resume_selector import CATEGORY_KEYWORDS, ResumeSelector

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Job Agent | Candidate Mission Control",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .score-badge-high {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.9rem;
    }
    .score-badge-med {
        background-color: #dbeafe;
        color: #1d4ed8;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.9rem;
    }
    .score-badge-low {
        background-color: #f1f5f9;
        color: #475569;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.9rem;
    }
    .skill-pill {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .missing-pill {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=5)
def load_data():
    """Load jobs, applications, and profile from database."""
    init_db()
    with get_db_session() as session:
        jobs = session.query(JobModel).all()
        job_list = []
        for j in jobs:
            matched = []
            missing = []
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

            job_list.append({
                "id": j.id,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "description": j.description,
                "url": j.url,
                "source": j.source,
                "employment_type": j.employment_type or "Full-time",
                "salary": j.salary or "Not disclosed",
                "status": j.status.value if hasattr(j.status, "value") else str(j.status),
                "match_score": j.match_score or 0.0,
                "deterministic_score": j.deterministic_score or 0.0,
                "ai_score": j.ai_score,
                "final_score": j.final_score or 0.0,
                "match_category": j.match_category or "UNSCORED",
                "matched_skills": matched,
                "missing_skills": missing,
                "match_reason": j.match_reason or "",
                "collected_at": j.collected_at,
            })

        apps = session.query(ApplicationModel).all()
        app_list = [
            {
                "id": a.id,
                "job_id": a.job_id,
                "company": a.company,
                "role": a.role,
                "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                "resume_used": a.resume_used,
                "applied_date": a.applied_date,
                "notes": a.notes,
            }
            for a in apps
        ]

    try:
        profile = load_user_profile()
    except Exception:
        profile = None

    return job_list, app_list, profile


# Fetch data
jobs_data, apps_data, profile = load_data()
selector = ResumeSelector()

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=100)
    st.title("AI Job Agent")
    st.caption("Human-in-the-Loop Career Assistant")

    if profile:
        st.markdown(f"**👤 Candidate:** {profile.name}")
        st.markdown(f"**🎓 Education:** SRIT ECE ({profile.education[0].graduation_year if profile.education else '2027'})")
        st.markdown(f"**📍 Location:** {profile.location}")
        st.caption(f"🎯 Target: {', '.join(profile.target_roles[:3])}...")
    
    st.divider()

    st.subheader("System Status")
    st.markdown(f"**🧠 LLM Provider:** `{settings.AI_PROVIDER}`")
    st.markdown(f"**🤖 Model:** `{settings.OLLAMA_MODEL}`")
    st.markdown(f"**📂 Database:** `{settings.DATABASE_URL.split('///')[-1]}`")
    st.markdown(f"**🛡️ Human Approval:** `{'Required' if settings.REQUIRE_HUMAN_APPROVAL else 'Automated'}`")

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ----------------------------------------------------------------------------
# HEADER & METRICS
# ----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎯 AI Job & Internship Application Control Center</div>', unsafe_allow_html=True)
st.caption("Automated job intelligence, tailored ATS resume matching, and candidate application tracker.")

total_jobs = len(jobs_data)
high_matches = sum(1 for j in jobs_data if j["match_score"] >= 70.0)
shortlisted = sum(1 for j in jobs_data if j["status"] == "SHORTLISTED")
applied_count = sum(1 for j in jobs_data if j["status"] == "APPLIED")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Discovered Jobs", total_jobs)
col2.metric("Target Matches (≥70%)", high_matches)
col3.metric("Shortlisted", shortlisted)
col4.metric("Applied", applied_count)
col5.metric("Active Model", settings.OLLAMA_MODEL.split(":")[0])

st.divider()


# ----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# ----------------------------------------------------------------------------
tab_jobs, tab_pipeline, tab_actions, tab_profile, tab_resumes = st.tabs([
    "🎯 Matched Opportunities",
    "📊 Application Pipeline",
    "⚡ Ingestion & AI Matching",
    "👤 Candidate Profile",
    "📄 Resume Tracks",
])


# ============================================================================
# TAB 1: MATCHED OPPORTUNITIES
# ============================================================================
with tab_jobs:
    # Filter Controls
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 2])
    with f_col1:
        search_query = st.text_input("🔍 Search Title or Company", "")
    with f_col2:
        status_options = ["ALL", "NEW", "MATCHED", "SHORTLISTED", "READY_TO_APPLY", "APPLIED", "REJECTED"]
        selected_status = st.selectbox("Status Filter", status_options, index=0)
    with f_col3:
        min_score_filter = st.slider("Minimum Match Score", 0, 100, 50, step=5)
    with f_col4:
        source_options = ["ALL", "greenhouse", "lever"]
        selected_source = st.selectbox("Source Platform", source_options, index=0)

    # Filter logic
    filtered_jobs = jobs_data
    if search_query:
        sq = search_query.lower()
        filtered_jobs = [j for j in filtered_jobs if sq in j["title"].lower() or sq in j["company"].lower()]
    if selected_status != "ALL":
        filtered_jobs = [j for j in filtered_jobs if j["status"] == selected_status]
    filtered_jobs = [j for j in filtered_jobs if j["match_score"] >= min_score_filter]
    if selected_source != "ALL":
        filtered_jobs = [j for j in filtered_jobs if j["source"] == selected_source]

    # Sort descending by match score
    filtered_jobs = sorted(filtered_jobs, key=lambda x: x["match_score"], reverse=True)

    st.markdown(f"**Showing {len(filtered_jobs)} of {total_jobs} Opportunities**")

    # Display Jobs
    for job in filtered_jobs[:50]:  # paginate top 50 in view
        score = job["match_score"]
        if score >= 75:
            score_badge = f'<span class="score-badge-high">⭐ Match: {score:.1f}%</span>'
        elif score >= 60:
            score_badge = f'<span class="score-badge-med">👍 Match: {score:.1f}%</span>'
        else:
            score_badge = f'<span class="score-badge-low">Match: {score:.1f}%</span>'

        with st.expander(f"{score_badge} **{job['title']}** at **{job['company']}** — {job['location']}", expanded=(score >= 75)):
            c_info1, c_info2 = st.columns([3, 1])
            with c_info1:
                st.markdown(f"**Employment Type:** {job['employment_type']} | **Source:** `{job['source']}` | **Current Status:** `{job['status']}`")
                
                # Recommendation Track
                rec = selector.recommend_resume(
                    title=job["title"],
                    description=job["description"],
                    matched_skills=job["matched_skills"],
                )
                st.markdown(f"**📄 Recommended Resume:** `{rec.category_name}` (`{rec.suggested_filename}`)")
                
                # Skills Display
                if job["matched_skills"]:
                    st.markdown("**Matched Candidate Skills:**")
                    skills_html = "".join([f'<span class="skill-pill">✓ {s}</span>' for s in job["matched_skills"]])
                    st.markdown(skills_html, unsafe_allow_html=True)

                if job["missing_skills"]:
                    st.markdown("**Missing / Desired Skills:**")
                    missing_html = "".join([f'<span class="missing-pill">! {s}</span>' for s in job["missing_skills"][:6]])
                    st.markdown(missing_html, unsafe_allow_html=True)

                if job["match_reason"]:
                    st.info(f"💡 **AI Match Analysis:** {job['match_reason']}")

                with st.expander("View Full Job Description"):
                    st.write(job["description"])

            with c_info2:
                st.markdown("### Actions")
                st.link_button("🌐 Open Job Link", job["url"], use_container_width=True)

                def update_status(job_id: int, new_status: ApplicationStatus):
                    with get_db_session() as s:
                        db_j = s.query(JobModel).filter_by(id=job_id).first()
                        if db_j:
                            db_j.status = new_status
                    st.cache_data.clear()
                    st.rerun()

                if job["status"] != "SHORTLISTED":
                    if st.button("⭐ Shortlist Job", key=f"shortlist_{job['id']}", use_container_width=True):
                        update_status(job["id"], ApplicationStatus.SHORTLISTED)

                if job["status"] != "READY_TO_APPLY":
                    if st.button("📋 Ready to Apply", key=f"ready_{job['id']}", use_container_width=True):
                        update_status(job["id"], ApplicationStatus.READY_TO_APPLY)

                if job["status"] != "APPLIED":
                    if st.button("🚀 Mark as Applied", key=f"applied_{job['id']}", use_container_width=True):
                        update_status(job["id"], ApplicationStatus.APPLIED)

                if job["status"] != "REJECTED":
                    if st.button("❌ Dismiss Job", key=f"reject_{job['id']}", use_container_width=True):
                        update_status(job["id"], ApplicationStatus.REJECTED)


# ============================================================================
# TAB 2: APPLICATION PIPELINE
# ============================================================================
with tab_pipeline:
    st.subheader("Candidate Application Workflow")
    st.caption("Human-in-the-loop progression from shortlist to interview.")

    stages = [
        ("Shortlisted", "SHORTLISTED", "⭐"),
        ("Ready to Apply", "READY_TO_APPLY", "📋"),
        ("Applied", "APPLIED", "🚀"),
        ("Interview Scheduled", "INTERVIEW", "🎙️"),
    ]

    p_cols = st.columns(len(stages))
    for idx, (label, st_val, emoji) in enumerate(stages):
        stage_jobs = [j for j in jobs_data if j["status"] == st_val]
        with p_cols[idx]:
            st.markdown(f"### {emoji} {label} ({len(stage_jobs)})")
            for sj in stage_jobs[:10]:
                st.info(f"**{sj['title']}**\n\n🏢 {sj['company']}\n\n📍 {sj['location']}\n\nScore: **{sj['match_score']:.1f}%**")


# ============================================================================
# TAB 3: INGESTION & MATCHING ENGINE
# ============================================================================
with tab_actions:
    st.subheader("Automation Control & Engine Triggers")
    
    act_col1, act_col2 = st.columns(2)

    with act_col1:
        st.markdown("### 📥 ATS Job Collection")
        st.write("Harvest new job openings directly from Greenhouse and Lever company career boards.")
        dry_run_collect = st.checkbox("Dry Run (Simulate without writing to DB)", value=False, key="dry_collect")

        if st.button("🚀 Start Collection Run", type="primary", use_container_width=True):
            with st.spinner("Connecting to Greenhouse & Lever APIs..."):
                try:
                    manager = CollectorManager()
                    stats = manager.collect_all(dry_run=dry_run_collect)
                    st.success(f"Discovered {stats['total_new_jobs']} new jobs!")
                    st.json(stats)
                    st.cache_data.clear()
                except Exception as err:
                    st.error(f"Collection failed: {err}")

    with act_col2:
        st.markdown("### 🧠 AI Hybrid Matching Engine")
        st.write("Run two-stage matching: Deterministic rule evaluation followed by Ollama LLM semantic scoring.")
        
        match_mode = st.radio("AI Mode", ["Hybrid (Deterministic + Ollama qwen3.5)", "Fast Deterministic Only"], index=1)
        ai_limit = st.slider("Max jobs for AI evaluation", 5, 50, 10)
        dry_run_match = st.checkbox("Dry Run (Do not save to DB)", value=False, key="dry_match")

        if st.button("⚡ Run Batch Matching Engine", type="primary", use_container_width=True):
            with st.spinner("Evaluating jobs against profile criteria..."):
                try:
                    m_manager = MatchingManager()
                    stats = m_manager.run_matching(
                        ai_limit=ai_limit,
                        dry_run=dry_run_match,
                        no_ai=(match_mode == "Fast Deterministic Only"),
                    )
                    st.success(f"Evaluated {stats['deterministic_evaluated']} jobs!")
                    st.json(stats)
                    st.cache_data.clear()
                except Exception as err:
                    st.error(f"Matching failed: {err}")


# ============================================================================
# TAB 4: CANDIDATE PROFILE
# ============================================================================
with tab_profile:
    st.subheader("Candidate Information & Preferences")
    if profile:
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown(f"**Name:** {profile.name}")
            st.markdown(f"**Email:** {profile.email}")
            st.markdown(f"**Phone:** {profile.phone}")
            st.markdown(f"**Location:** {profile.location}")
            st.markdown(f"**LinkedIn:** [{profile.linkedin}]({profile.linkedin})")
            st.markdown(f"**GitHub:** [{profile.github}]({profile.github})")
            st.markdown(f"**Summary:** {profile.summary}")

        with p_c2:
            st.markdown("### Target Roles")
            for r in profile.target_roles:
                st.markdown(f"- {r}")

            st.markdown("### Technical Skills")
            skills_html = "".join([f'<span class="skill-pill">{s}</span>' for s in profile.skills])
            st.markdown(skills_html, unsafe_allow_html=True)

        st.divider()
        st.markdown("### Key Projects")
        for p in profile.projects:
            with st.container(border=True):
                st.markdown(f"**{p.title}**")
                st.write(p.description)
                st.caption(f"Technologies: {', '.join(p.technologies)}")


# ============================================================================
# TAB 5: RESUME TRACKS
# ============================================================================
with tab_resumes:
    st.subheader("Candidate Resume Variations")
    st.caption("Tailored resume versions mapped to candidate domains.")

    available_files = selector.list_available_resumes()

    r_cols = st.columns(4)
    for idx, (cat_key, cat_data) in enumerate(CATEGORY_KEYWORDS.items()):
        with r_cols[idx]:
            with st.container(border=True):
                st.markdown(f"#### {cat_data['name']}")
                st.caption(f"Folder: `resumes/{cat_key}/`")
                files = available_files.get(cat_key, [])
                if files:
                    st.success(f"📄 Found: `{files[0]}`")
                else:
                    st.warning(f"No PDF yet. Expected: `{cat_data['default_file']}`")

                st.markdown("**Target Keywords:**")
                st.write(", ".join(cat_data["keywords"][:6]))
