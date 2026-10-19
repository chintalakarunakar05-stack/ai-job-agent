"""Database session and table initialization management.

Provides SQLAlchemy engine, session maker, context managers,
and database bootstrap utilities supporting SQLite and PostgreSQL.
"""

import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.models import Base

logger = logging.getLogger(__name__)

# Configure engine arguments
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite requires check_same_thread=False for multi-threaded access (e.g. FastAPI / Streamlit)
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # Set to True for SQL query debugging
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def _upgrade_sqlite_schema() -> None:
    """Safely apply schema migrations for new columns without dropping existing data."""
    if not settings.DATABASE_URL.startswith("sqlite"):
        return

    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            # Check existing columns in jobs table
            columns_info = conn.execute(text("PRAGMA table_info(jobs);")).fetchall()
            existing_columns = {col[1] for col in columns_info}
            if not existing_columns:
                return

            new_columns = [
                ("deterministic_score", "FLOAT DEFAULT 0.0"),
                ("ai_score", "FLOAT"),
                ("final_score", "FLOAT DEFAULT 0.0"),
                ("match_category", "VARCHAR(64)"),
                ("matched_skills", "TEXT"),
                ("ai_analysis", "TEXT"),
                ("analyzed_at", "DATETIME"),
            ]

            for col_name, col_type in new_columns:
                if col_name not in existing_columns:
                    logger.info(f"Applying schema migration: adding column '{col_name}' to 'jobs' table.")
                    conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {col_name} {col_type};"))
            conn.commit()
        except Exception as exc:
            logger.warning(f"Schema upgrade check encountered warning: {exc}")


def init_db() -> None:
    """Initialize database tables according to defined ORM models."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    _upgrade_sqlite_schema()
    logger.info("Database tables initialized successfully.")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session scope."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Database session error, rolled back: {exc}")
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
