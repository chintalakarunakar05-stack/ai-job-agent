FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-cached separately from app code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application source
COPY . .

# Ensure runtime data directories exist inside the container
# SQLite DB and logs will be written here at runtime
RUN mkdir -p data/logs resumes

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck so Railway knows when the app is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Dynamic PORT support for Railway / Render / Fly.io
CMD ["sh", "-c", "streamlit run app/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
