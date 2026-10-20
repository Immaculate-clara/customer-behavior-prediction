"""Score every customer and store the results in PostgreSQL.

Run with:  python -m src.data.store_scores
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, func, insert, select
from sqlalchemy.exc import OperationalError

from src.utils.config import get_config_value
from src.utils.database import (
    customer_scores,
    get_engine,
    init_db,
    scoring_runs,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run() -> None:
    from api.portfolio import Portfolio
    from api.services import PredictionService

    service = PredictionService()
    table = Portfolio(service).table.copy()

    now = datetime.now(timezone.utc)
    table["customer_id"] = table["customer_id"].astype(int)
    table["predicted_churn"] = table["predicted_churn"].astype(bool)
    table["segment_id"] = table["segment_id"].astype(int)
    table["scored_at"] = now

    engine = get_engine()
    try:
        init_db(engine)
        with engine.begin() as conn:
            conn.execute(delete(customer_scores))
            table.to_sql(
                "customer_scores",
                conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
            conn.execute(
                insert(scoring_runs).values(
                    created_at=now,
                    model_name=str(get_config_value("api.model_name", "xgboost")),
                    customers=int(len(table)),
                    high_risk_customers=int((table["risk_level"] == "high").sum()),
                )
            )
    except OperationalError as exc:
        raise SystemExit(
            "Cannot connect to PostgreSQL. Start it with 'docker compose up -d db' "
            f"and check DATABASE_URL in .env.\n{exc.orig}"
        ) from exc

    with engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(customer_scores)).scalar_one()
        by_risk = conn.execute(
            select(customer_scores.c.risk_level, func.count())
            .group_by(customer_scores.c.risk_level)
            .order_by(customer_scores.c.risk_level)
        ).all()
    logger.info("Stored %s customers in PostgreSQL", f"{total:,}")
    for level, count in by_risk:
        logger.info("  %-6s %s", level, f"{count:,}")


if __name__ == "__main__":
    run()