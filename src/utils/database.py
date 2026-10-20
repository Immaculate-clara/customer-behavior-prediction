"""PostgreSQL connection and table definitions."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import Engine

from src.utils.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATABASE_URL = "postgresql+psycopg2://cbp:cbp_password@localhost:5432/customer_behavior"

metadata = MetaData()

customer_scores = Table(
    "customer_scores",
    metadata,
    Column("customer_id", Integer, primary_key=True),
    Column("churn_probability", Float, nullable=False),
    Column("predicted_churn", Boolean, nullable=False),
    Column("risk_level", String(10), nullable=False),
    Column("predicted_revenue", Float, nullable=False),
    Column("segment_id", Integer, nullable=False),
    Column("segment_name", String(60), nullable=False),
    Column("recency_days", Float),
    Column("frequency", Float),
    Column("monetary_total", Float),
    Column("priority_score", Float),
    Column("scored_at", DateTime(timezone=True), nullable=False),
)
Index("ix_customer_scores_risk", customer_scores.c.risk_level)
Index("ix_customer_scores_segment", customer_scores.c.segment_name)

scoring_runs = Table(
    "scoring_runs",
    metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model_name", String(50)),
    Column("customers", Integer, nullable=False),
    Column("high_risk_customers", Integer, nullable=False),
)


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True)


def init_db(engine: Engine | None = None) -> None:
    """Create the tables if they do not exist yet."""
    metadata.create_all(engine or get_engine())