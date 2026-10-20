"""Portfolio-level scores for the dashboard.

At start-up every customer in the feature table is scored once (churn probability,
expected revenue and segment). The dashboard endpoints then only filter and
aggregate this in-memory table, so they respond instantly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RISK_LEVELS = ("low", "medium", "high")
SORT_COLUMNS = {
    "priority": "priority_score",
    "churn_probability": "churn_probability",
    "predicted_revenue": "predicted_revenue",
    "monetary_total": "monetary_total",
}
HISTOGRAM_BINS = 20


class Portfolio:
    """Scored customer table with summary and filtering helpers."""

    def __init__(self, service) -> None:
        frame = service.features.reset_index()
        ids = frame["customer_id"].astype(int).tolist()
        features = frame.drop(columns="customer_id")

        churn = pd.DataFrame(service.churn(features, ids))
        clv = pd.DataFrame(service.clv(features, ids))
        segment = pd.DataFrame(service.segment(features, ids))

        table = pd.DataFrame(
            {
                "customer_id": ids,
                "churn_probability": churn["churn_probability"].to_numpy(),
                "predicted_churn": churn["predicted_churn"].to_numpy(),
                "risk_level": churn["risk_level"].to_numpy(),
                "predicted_revenue": clv["predicted_revenue"].to_numpy(),
                "segment_id": segment["segment_id"].to_numpy(),
                "segment_name": segment["segment_name"].to_numpy(),
                "recency_days": features["recency_days"].to_numpy(),
                "frequency": features["frequency"].to_numpy(),
                "monetary_total": features["monetary_total"].to_numpy(),
            }
        )
        # Retention priority: likely to leave AND valuable so far.
        table["priority_score"] = table["churn_probability"] * table["monetary_total"].clip(lower=0)
        self.table = table
        logger.info("Portfolio scored for %s customers", f"{len(table):,}")

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        table = self.table
        counts = table["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
        high = table[table["risk_level"] == "high"]

        grouped = table.groupby(["segment_id", "segment_name"])
        segments = grouped.agg(
            customers=("customer_id", "size"),
            avg_churn_probability=("churn_probability", "mean"),
            expected_revenue=("predicted_revenue", "sum"),
            historical_spend=("monetary_total", "sum"),
        ).reset_index()
        segments["high_risk_share"] = (
            grouped["risk_level"].apply(lambda s: float((s == "high").mean())).to_numpy()
        )

        edges = np.linspace(0.0, 1.0, HISTOGRAM_BINS + 1)
        bin_counts, _ = np.histogram(table["churn_probability"], bins=edges)

        return {
            "customers": int(len(table)),
            "avg_churn_probability": float(table["churn_probability"].mean()),
            "high_risk_customers": int(len(high)),
            "high_risk_share": float(len(high) / len(table)) if len(table) else 0.0,
            "expected_revenue": float(table["predicted_revenue"].sum()),
            "high_risk_historical_spend": float(high["monetary_total"].clip(lower=0).sum()),
            "risk_distribution": {level: int(counts[level]) for level in RISK_LEVELS},
            "churn_probability_histogram": [
                {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(bin_counts[i])}
                for i in range(HISTOGRAM_BINS)
            ],
            "segments": [
                {
                    "segment_id": int(row.segment_id),
                    "segment": str(row.segment_name),
                    "customers": int(row.customers),
                    "avg_churn_probability": float(row.avg_churn_probability),
                    "high_risk_share": float(row.high_risk_share),
                    "expected_revenue": float(row.expected_revenue),
                    "historical_spend": float(row.historical_spend),
                }
                for row in segments.sort_values("segment_id").itertuples()
            ],
        }

    def customers(
        self,
        risk_level: str | None = None,
        segment: str | None = None,
        sort_by: str = "priority",
        limit: int = 50,
    ) -> dict[str, Any]:
        if sort_by not in SORT_COLUMNS:
            raise ValueError(f"sort_by must be one of {sorted(SORT_COLUMNS)}")
        table = self.table
        if risk_level is not None:
            table = table[table["risk_level"] == risk_level]
        if segment is not None:
            table = table[table["segment_name"] == segment]
        top = table.sort_values(SORT_COLUMNS[sort_by], ascending=False).head(limit)
        rows = [
            {
                "customer_id": int(r.customer_id),
                "churn_probability": float(r.churn_probability),
                "risk_level": str(r.risk_level),
                "predicted_churn": bool(r.predicted_churn),
                "predicted_revenue": float(r.predicted_revenue),
                "segment_id": int(r.segment_id),
                "segment_name": str(r.segment_name),
                "recency_days": float(r.recency_days),
                "frequency": float(r.frequency),
                "monetary_total": float(r.monetary_total),
                "priority_score": float(r.priority_score),
            }
            for r in top.itertuples()
        ]
        return {"total": int(len(table)), "returned": len(rows), "customers": rows}