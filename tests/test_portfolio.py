"""Tests for the portfolio (dashboard) scoring logic and endpoints."""
from __future__ import annotations

import pandas as pd
import pytest

from api.portfolio import Portfolio


class FakeScoringService:
    """Minimal stand-in exposing what ``Portfolio`` needs."""

    def __init__(self) -> None:
        self.features = pd.DataFrame(
            {
                "recency_days": [10.0, 300.0, 200.0, 5.0],
                "frequency": [5.0, 1.0, 2.0, 9.0],
                "monetary_total": [1000.0, 50.0, 400.0, 3000.0],
            },
            index=pd.Index([1, 2, 3, 4], name="customer_id"),
        )

    @staticmethod
    def _level(p):
        return "low" if p < 0.33 else "medium" if p < 0.66 else "high"

    def churn(self, X, ids):
        probabilities = [0.1, 0.9, 0.5, 0.05]
        return [
            {"customer_id": cid, "churn_probability": p, "predicted_churn": p >= 0.5, "risk_level": self._level(p)}
            for cid, p in zip(ids, probabilities)
        ]

    def clv(self, X, ids):
        return [{"customer_id": cid, "predicted_revenue": r} for cid, r in zip(ids, [100.0, 5.0, 40.0, 500.0])]

    def segment(self, X, ids):
        labels = [(0, "Champions"), (1, "Lost"), (1, "Lost"), (0, "Champions")]
        return [{"customer_id": cid, "segment_id": g, "segment_name": n} for cid, (g, n) in zip(ids, labels)]


@pytest.fixture()
def portfolio():
    return Portfolio(FakeScoringService())


def test_summary_totals(portfolio):
    summary = portfolio.summary()
    assert summary["customers"] == 4
    assert summary["risk_distribution"] == {"low": 2, "medium": 1, "high": 1}
    assert summary["high_risk_customers"] == 1
    assert summary["high_risk_share"] == pytest.approx(0.25)
    assert summary["high_risk_historical_spend"] == 50.0
    assert summary["expected_revenue"] == pytest.approx(645.0)
    assert sum(item["count"] for item in summary["churn_probability_histogram"]) == 4


def test_summary_segments(portfolio):
    segments = {item["segment"]: item for item in portfolio.summary()["segments"]}
    assert segments["Lost"]["customers"] == 2
    assert segments["Lost"]["high_risk_share"] == pytest.approx(0.5)
    assert segments["Champions"]["high_risk_share"] == 0.0


def test_default_sort_is_retention_priority(portfolio):
    # priority = churn probability x past spend: 200, 150, 100, 45
    ids = [row["customer_id"] for row in portfolio.customers()["customers"]]
    assert ids == [3, 4, 1, 2]


def test_filters_and_limit(portfolio):
    assert [row["customer_id"] for row in portfolio.customers(risk_level="high")["customers"]] == [2]
    result = portfolio.customers(segment="Champions", limit=1)
    assert result["total"] == 2
    assert result["returned"] == 1


def test_invalid_sort_is_rejected(portfolio):
    with pytest.raises(ValueError):
        portfolio.customers(sort_by="unknown")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import create_app

    monkeypatch.setattr("api.main.PredictionService", FakeScoringService)
    with TestClient(create_app()) as test_client:
        yield test_client


def test_summary_endpoint(client):
    body = client.get("/portfolio/summary").json()
    assert body["customers"] == 4
    assert body["risk_distribution"]["high"] == 1
    assert len(body["churn_probability_histogram"]) == 20


def test_customers_endpoint_filters(client):
    body = client.get("/portfolio/customers", params={"risk_level": "high"}).json()
    assert body["total"] == 1
    assert body["customers"][0]["customer_id"] == 2

    ranked = client.get("/portfolio/customers", params={"sort_by": "monetary_total", "limit": 2}).json()
    assert [row["customer_id"] for row in ranked["customers"]] == [4, 1]


def test_customers_endpoint_validates_input(client):
    assert client.get("/portfolio/customers", params={"risk_level": "extreme"}).status_code == 422
    assert client.get("/portfolio/customers", params={"limit": 0}).status_code == 422
    assert client.get("/portfolio/customers", params={"sort_by": "unknown"}).status_code == 422


def test_model_comparison_reads_saved_tables(client, monkeypatch, tmp_path):
    pd.DataFrame({"model": ["xgboost", "svm"], "test_roc_auc": [0.82, None]}).to_csv(
        tmp_path / "churn_model_comparison.csv", index=False
    )
    monkeypatch.setattr("api.routers.portfolio.resolve_path", lambda *_args, **_kwargs: tmp_path)
    body = client.get("/models/comparison").json()
    assert body["churn"][0] == {"model": "xgboost", "test_roc_auc": 0.82}
    assert body["churn"][1]["test_roc_auc"] is None
    assert body["deep_learning"] == []
    assert body["clv"] == []