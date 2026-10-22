"""Tests for the PostgreSQL storage layer, run against a temporary SQLite database."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, func, inspect, select  # noqa: E402

from src.data import store_scores  # noqa: E402
from src.utils import database  # noqa: E402


@pytest.fixture()
def engine(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture(autouse=True)
def fresh_engine_cache():
    database.get_engine.cache_clear()
    yield
    database.get_engine.cache_clear()


def test_database_url_defaults_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database.get_database_url() == database.DEFAULT_DATABASE_URL

    monkeypatch.setenv("DATABASE_URL", "sqlite:///custom.db")
    assert database.get_database_url() == "sqlite:///custom.db"


def test_engine_is_created_once(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'engine.db'}")
    assert database.get_engine() is database.get_engine()


def test_init_db_creates_both_tables(engine):
    database.init_db(engine)
    assert set(inspect(engine).get_table_names()) == {"customer_scores", "scoring_runs"}
    database.init_db(engine)  # safe to run twice


def test_init_db_uses_the_default_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'default.db'}")
    database.init_db()
    assert "customer_scores" in inspect(database.get_engine()).get_table_names()


# ------------------------------------------------------------------ store_scores
def _portfolio_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "churn_probability": [0.9, 0.7, 0.4, 0.1],
            "predicted_churn": [True, True, False, False],
            "risk_level": ["high", "high", "medium", "low"],
            "predicted_revenue": [10.0, 20.0, 30.0, 40.0],
            "segment_id": [0, 0, 1, 2],
            "segment_name": ["Lost Customers", "Lost Customers", "At Risk", "Champions"],
            "recency_days": [300.0, 250.0, 100.0, 5.0],
            "frequency": [1.0, 2.0, 5.0, 20.0],
            "monetary_total": [100.0, 200.0, 900.0, 5000.0],
            "priority_score": [90.0, 140.0, 360.0, 500.0],
        }
    )


@pytest.fixture()
def fake_scoring(monkeypatch, engine):
    class FakeService:
        pass

    class FakePortfolio:
        def __init__(self, service):
            self.table = _portfolio_table()

    monkeypatch.setattr("api.services.PredictionService", FakeService)
    monkeypatch.setattr("api.portfolio.Portfolio", FakePortfolio)
    monkeypatch.setattr(store_scores, "get_engine", lambda: engine)


def test_scores_are_stored_and_summarised(fake_scoring, engine):
    store_scores.run()

    with engine.connect() as conn:
        stored = conn.execute(select(func.count()).select_from(database.customer_scores)).scalar_one()
        run = conn.execute(select(database.scoring_runs)).one()
    assert stored == 4
    assert run.customers == 4 and run.high_risk_customers == 2


def test_each_run_replaces_the_scores_and_logs_a_run(fake_scoring, engine):
    store_scores.run()
    store_scores.run()

    with engine.connect() as conn:
        stored = conn.execute(select(func.count()).select_from(database.customer_scores)).scalar_one()
        runs = conn.execute(select(func.count()).select_from(database.scoring_runs)).scalar_one()
    assert stored == 4 and runs == 2


def test_unreachable_database_gives_a_helpful_message(fake_scoring, monkeypatch):
    pytest.importorskip("psycopg2")
    unreachable = create_engine("postgresql+psycopg2://user:pass@127.0.0.1:1/missing")
    monkeypatch.setattr(store_scores, "get_engine", lambda: unreachable)

    with pytest.raises(SystemExit, match="Cannot connect to PostgreSQL"):
        store_scores.run()