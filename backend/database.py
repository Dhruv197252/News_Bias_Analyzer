"""
Database — PostgreSQL via psycopg2 + SQLAlchemy sync
------------------------------------------------------
Render free tier safe:
  • Synchronous (no async complexity)
  • Connection pool sized for 512 MB RAM
  • Auto-creates tables on startup (no Alembic migrations needed)
"""

import logging
import os
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Text, DateTime, JSON
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
import uuid
from datetime import datetime, timezone

from backend.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────

def _make_engine():
    db_url = settings.DATABASE_URL

    # PostgreSQL (Neon / Render)
    if db_url.startswith("postgresql"):
        return create_engine(
            db_url,
            pool_size       = 3,      # Keep low for 512 MB RAM limit
            max_overflow    = 2,
            pool_timeout    = 30,
            pool_recycle    = 1800,   # Recycle connections every 30 min
            pool_pre_ping   = True,   # Detect stale connections
            echo            = settings.DEBUG,
        )

    # SQLite (local dev fallback)
    return create_engine(
        db_url,
        connect_args = {"check_same_thread": False},
        echo         = settings.DEBUG,
    )


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── ORM Models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _uuid_default():
    return str(uuid.uuid4())


class Analysis(Base):
    __tablename__ = "analyses"

    id               = Column(String(36), primary_key=True, default=_uuid_default)
    url              = Column(Text,    nullable=True)
    domain           = Column(String(255), nullable=True, index=True)
    headline         = Column(Text,    nullable=True)
    body_preview     = Column(Text,    nullable=True)
    composite_score  = Column(Float,   nullable=True)
    verdict          = Column(String(100), nullable=True)
    ml_probability   = Column(Float,   nullable=True)
    ml_label         = Column(Integer, nullable=True)   # 0 = Neutral, 1 = Biased
    engine_scores    = Column(JSON,    nullable=True)   # All engine scores as JSON
    genai_explanation = Column(Text,   nullable=True)
    genai_rewrite    = Column(Text,    nullable=True)
    genai_summary    = Column(Text,    nullable=True)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    analysis_version = Column(String(20), default="v2.0")


class DomainStat(Base):
    __tablename__ = "domain_stats"

    domain               = Column(String(255), primary_key=True)
    total_articles       = Column(Integer, default=0)
    avg_bias_score       = Column(Float,   default=0.0)
    mbfc_leaning         = Column(String(100), nullable=True)
    mbfc_factual_reporting = Column(String(100), nullable=True)
    country              = Column(String(10), nullable=True)
    last_updated         = Column(DateTime, nullable=True)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist. Called at app startup."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized.")
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")
        raise


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
