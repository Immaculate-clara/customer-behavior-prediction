"""API tests: routing, validation and error handling.

A small fake service replaces the real models, so these tests need no trained
artifacts and run in a few seconds.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import CustomerNotFound, InvalidFeatures


class FakeService:
    """Deterministic stand-in for ``PredictionService``."""

    feature_columns = ["recency_days", "frequency"]
    known = {1: (50.0, 8.0), 2: (400.0, 1.0)}

    def __init__(self) -> None:
        pass

    def health(self):
        return {
            "status": "ok",
            "churn_model": "fake",
            "clv_model": "fake",
            "customers_available": len(self.known),
            "feature_columns": self.feature_columns,
        }

    def resolve(self, customer_id, features):
        if customer_id is not None:
            return self.resolve_many([customer_id])
        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            raise InvalidFeatures(f"Missing features: {missing}")
        return [(features["recency_days"], features["frequency"])], [None]

    def resolve_many(self, customer_ids):
        unknown = [cid for cid in customer_ids if cid not in self.known]
        if unknown:
            raise CustomerNotFound(f"Unknown customer_id(s): {unknown}")
        return [self.known[cid] for cid in customer_ids], list(customer_ids)

    @staticmethod
    def _level(probability):
        return "low" if probability < 0.33 else "medium" if probability < 0.66 else "high"

    def churn(self, X, ids):
        results = []
        for cid, (recency, _) in zip(ids, X):
            p = min(recency / 500.0, 1.0)
            results.append(
                {"customer_id": cid, "churn_probability": p, "predicted_churn": p >= 0.5,
                 "risk_level": self._level(p), "threshold": 0.5, "model": "fake"}
            )
        return results

    def clv(self, X, ids):
        return [{"customer_id": cid, "predicted_revenue": 10.0 * freq, "model": "fake"} for cid, (_, freq) in zip(ids, X)]

    def segment(self, X, ids):
        return [{"customer_id": cid, "segment_id": 0, "segment_name": "Champions"} for cid in ids]

    def profiles(self, X, ids):
        return [
            {"customer_id": cid, "churn": a, "clv": b, "segment": c}
            for cid, a, b, c in zip(ids, self.churn(X, ids), self.clv(X, ids), self.segment(X, ids))
        ]

    def explain(self, X, ids, top_n):
        features = [
            {"feature": "recency_days", "value": 50.0, "shap_value": 0.4, "effect": "increases churn risk"},
            {"feature": "frequency", "value": 8.0, "shap_value": -0.2, "effect": "reduces churn risk"},
        ]
        return [
            {"customer_id": cid, "churn_probability": item["churn_probability"], "top_features": features[:top_n]}
            for cid, item in zip(ids, self.churn(X, ids))
        ]

    def segment_summaries(self):
        return [
            {"segment_id": 0, "segment": "Champions", "customers": 10, "share_pct": 50.0,
             "churn_rate": 0.2, "avg_future_revenue": 900.0, "averages": {"recency_days": 30.0}}
        ]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr("api.main.PredictionService", FakeService)
    monkeypatch.setattr("api.main.Portfolio", lambda service: object())  # not needed by these tests
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["feature_columns"] == ["recency_days", "frequency"]


def test_churn_by_customer_id(client):
    body = client.post("/predict/churn", json={"customer_id": 2}).json()
    assert body["customer_id"] == 2
    assert body["risk_level"] == "high"
    assert body["predicted_churn"] is True
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_churn_by_features(client):
    response = client.post("/predict/churn", json={"features": {"recency_days": 20, "frequency": 5}})
    assert response.status_code == 200
    assert response.json()["customer_id"] is None
    assert response.json()["risk_level"] == "low"


def test_unknown_customer_returns_404(client):
    response = client.post("/predict/churn", json={"customer_id": 999})
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_missing_features_return_422(client):
    response = client.post("/predict/churn", json={"features": {"recency_days": 20}})
    assert response.status_code == 422
    assert "frequency" in response.json()["detail"]


@pytest.mark.parametrize("payload", [{}, {"customer_id": 1, "features": {"recency_days": 1, "frequency": 1}}])
def test_exactly_one_input_source_is_required(client, payload):
    assert client.post("/predict/churn", json=payload).status_code == 422


def test_clv_and_segment(client):
    assert client.post("/predict/clv", json={"customer_id": 1}).json()["predicted_revenue"] == 80.0
    assert client.post("/predict/segment", json={"customer_id": 1}).json()["segment_name"] == "Champions"


def test_batch(client):
    body = client.post("/predict/batch", json={"customer_ids": [1, 2]}).json()
    assert body["count"] == 2
    assert [item["customer_id"] for item in body["results"]] == [1, 2]
    assert set(body["results"][0]) == {"customer_id", "churn", "clv", "segment"}


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customer_ids": []}).status_code == 422


def test_customer_profile(client):
    body = client.get("/customers/1").json()
    assert body["churn"]["risk_level"] == "low"
    assert body["clv"]["predicted_revenue"] == 80.0
    assert client.get("/customers/999").status_code == 404


def test_segments(client):
    body = client.get("/segments").json()
    assert body[0]["segment"] == "Champions"
    assert body[0]["churn_rate"] == 0.2


def test_explain(client):
    body = client.post("/explain/churn", json={"customer_id": 1, "top_n": 1}).json()
    assert len(body["top_features"]) == 1
    assert body["top_features"][0]["feature"] == "recency_days"


def test_explain_rejects_invalid_top_n(client):
    assert client.post("/explain/churn", json={"customer_id": 1, "top_n": 0}).status_code == 422