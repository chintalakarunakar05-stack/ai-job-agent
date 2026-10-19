"""Configuration settings for the AI Job & Internship Application Agent.

Loads environment variables from .env using Pydantic BaseSettings.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration parameters."""

    # Project Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = DATA_DIR / "logs"
    RESUMES_DIR: Path = BASE_DIR / "resumes"

    # Application & Environment
    APP_ENV: str = Field(default="development", description="Environment mode: development | production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging verbosity: DEBUG, INFO, WARNING, ERROR")
    PROFILE_PATH: Path = Field(default=Path("./data/profile.json"), description="Path to user profile JSON file")

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./data/jobs.db",
        description="Database connection string (SQLite by default, PostgreSQL ready)",
    )

    # AI Provider Settings
    AI_PROVIDER: str = Field(default="ollama", description="Active AI provider: ollama | openai | mock")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama API server URL")
    OLLAMA_MODEL: str = Field(default="qwen3.5:0.8b", description="Local model name for matching and generation")
    AI_MATCH_LIMIT: int = Field(default=50, description="Max number of top deterministic matches to evaluate with AI")
    OLLAMA_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Maximum wait time in seconds for a single Ollama inference request. Increase if your hardware is slow.",
    )
    
    # Optional Third-Party Providers
    OPENAI_API_KEY: str | None = Field(default=None, description="Optional OpenAI API key")
    OPENAI_BASE_URL: str | None = Field(default=None, description="Optional OpenAI-compatible API base URL")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="Model name for OpenAI-compatible API")

    # Browser Automation & Safety Safeguards
    HEADLESS_BROWSER: bool = Field(default=False, description="Whether to run browser in headless mode")
    AUTOMATION_TIMEOUT_MS: int = Field(default=30000, description="Browser element wait timeout in milliseconds")
    REQUIRE_HUMAN_APPROVAL: bool = Field(
        default=True,
        description="CRITICAL: Forces pause for human review and explicit approval before any form submission",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global singleton settings instance
settings = Settings()

# Ensure mandatory directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESUMES_DIR.mkdir(parents=True, exist_ok=True)
