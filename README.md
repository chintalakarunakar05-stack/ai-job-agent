# AI Job & Internship Application Agent 🎯

An intelligent, human-in-the-loop application assistant that automates job discovery from major ATS platforms (Greenhouse, Lever), performs two-stage hybrid candidate matching (deterministic rules + local LLM semantic evaluation via Ollama), and provides a Streamlit control center for reviewing and tracking applications.

---

## 🌟 Key Features

- **Automated ATS Harvesting:** Ingests live job postings from Greenhouse and Lever career boards with URL deduplication and schema normalization.
- **Two-Stage Hybrid Matching Engine:**
  - **Stage 1 (Deterministic):** Evaluates role alignment, technical skills, education compatibility, experience level, location suitability, and employment type.
  - **Stage 2 (Local LLM Evaluation):** Evaluates top qualifying jobs with Ollama (`qwen3.5:0.8b` or compatible) for deep semantic requirement fit and tailored advice.
  - **Hard Eligibility Gate:** Automatically disqualifies senior/lead roles and non-technical (sales/marketing/HR) positions for fresher and intern profiles.
- **Multi-Track Resume Recommender:** Suggests tailored resume tracks (`ai_ml`, `embedded`, `software`, `general`) and highlights matching keywords.
- **Mission Control Dashboard:** Interactive Streamlit web interface with job search/filtering, match score badges, skill breakdowns, and an application pipeline tracker.
- **RESTful API Backend:** High-performance FastAPI server providing endpoints for jobs, statistics, candidate profiles, and batch evaluation triggers.

---

## 🏗️ Architecture

```
AI Job & Internship Application Agent/
├── app/
│   └── dashboard.py          # Streamlit UI Control Center
├── src/
│   ├── ai/                   # Deterministic & LLM matchers (Ollama)
│   ├── collectors/           # Greenhouse & Lever job harvesters
│   ├── resumes/              # Resume selection & keyword analyzer
│   ├── config.py             # Pydantic settings & environment configs
│   ├── database.py           # SQLite connection and migration utilities
│   ├── main.py               # FastAPI backend API entrypoint
│   └── models.py             # SQLAlchemy ORM models & Pydantic schemas
├── data/
│   ├── profile.json          # Active candidate profile & preferences
│   └── sources.json          # Company ATS board sources
├── tests/                    # Comprehensive unit & integration test suite (65 tests)
├── Dockerfile                # Container deployment spec
├── Procfile                  # Platform-as-a-Service runner spec (Railway / Render)
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) (optional for local LLM evaluation; deterministic matching works standalone)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/OnlyNewApps/ai-job-agent.git
cd ai-job-agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Services

**Start the Streamlit Web Dashboard:**
```powershell
streamlit run app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

**Start the FastAPI Backend API:**
```powershell
uvicorn src.main:app --reload --port 8000
```
API Documentation & Swagger UI will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 🧪 Testing

Run the full test suite with pytest:
```powershell
pytest -q
```

---

## ☁️ Cloud Deployment

The repository includes ready-to-use configuration files for cloud hosting:
- **Streamlit Community Cloud:** Point directly to `app/dashboard.py`.
- **Railway.app / Render.com:** Uses [`Dockerfile`](Dockerfile) or [`Procfile`](Procfile) for automatic deployment.

---

## 🔒 Safety & Ethics

- **Human-in-the-Loop:** Automated applications require explicit human approval before any form submission.
- **Zero Secret Exposure:** Credentials and private keys are isolated via environment variables (`.env`) and excluded by `.gitignore`.
