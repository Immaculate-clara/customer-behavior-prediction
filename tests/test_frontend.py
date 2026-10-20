"""Tests for the dashboard's non-Streamlit code: API client, business rules and charts."""
from __future__ import annotations

import pandas as pd
import pytest
import requests

import api_client
import insights


# --------------------------------------------------------------------------- #
# Business rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("risk", "segment", "expected"),
    [
        ("high", "Champions", "Top retention priority"),
        ("high", "Lost Customers", "win-back"),
        ("medium", "At Risk", "Keep the customer warm"),
        ("low", "Champions", "reward loyalty"),
        ("low", "Lost Customers", "No retention action"),
    ],
)
def test_recommendations(risk, segment, expected):
    assert expected.lower() in insights.recommend(risk, segment).lower()


def test_money_format():
    assert insights.money(12345.6) == "£12,346"


def test_feature_labels():
    assert insights.pretty_feature("recency_days") == "Days since last order"
    assert insights.pretty_feature("affinity_kitchen_dining") == "Spend share: Kitchen Dining"
    assert insights.pretty_feature("some_new_feature") == "Some new feature"


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_client_returns_json(monkeypatch):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(method=method, url=url, **kwargs)
        return FakeResponse(payload={"status": "ok"})

    monkeypatch.setattr(requests, "request", fake_request)
    assert api_client.health() == {"status": "ok"}
    assert seen["method"] == "GET"
    assert seen["url"].endswith("/health")


def test_client_sends_filters(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "request", lambda method, url, **kw: seen.update(kw) or FakeResponse(payload={}))
    api_client.portfolio_customers(risk_level="high", segment="Champions", sort_by="priority", limit=10)
    assert seen["params"] == {"sort_by": "priority", "limit": 10, "risk_level": "high", "segment": "Champions"}


def test_client_reports_api_errors(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *a, **k: FakeResponse(404, {"detail": "Unknown customer_id(s): [1]"}))
    with pytest.raises(api_client.ApiError, match="404.*Unknown customer_id"):
        api_client.customer_profile(1)


def test_client_reports_unreachable_api(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "request", refuse)
    with pytest.raises(api_client.ApiError, match="Cannot reach the API"):
        api_client.health()


# --------------------------------------------------------------------------- #
# Charts (skipped when plotly is not installed)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def charts():
    pytest.importorskip("plotly")
    import charts as module

    return module


SEGMENTS = [
    {"segment_id": 0, "segment": "Champions", "customers": 10, "share_pct": 50.0, "avg_churn_probability": 0.2,
     "high_risk_share": 0.1, "expected_revenue": 900.0, "historical_spend": 5000.0,
     "averages": {"recency_days": 30.0, "frequency": 12.0}},
    {"segment_id": 1, "segment": "Lost Customers", "customers": 10, "share_pct": 50.0, "avg_churn_probability": 0.9,
     "high_risk_share": 0.8, "expected_revenue": 50.0, "historical_spend": 300.0,
     "averages": {"recency_days": 400.0, "frequency": 1.0}},
]


def test_segment_charts(charts):
    assert len(charts.segment_churn_chart(SEGMENTS).data) == 1
    assert len(charts.segment_revenue_chart(SEGMENTS).data) == 1
    assert len(charts.segment_size_chart(SEGMENTS).data) == 1
    heatmap = charts.segment_heatmap(SEGMENTS, ["recency_days", "frequency", "not_a_column"])
    z_values = heatmap.data[0].z
    assert len(z_values) == 2 and len(z_values[0]) == 2  # 2 segments x 2 valid features


def test_risk_charts(charts):
    assert list(charts.risk_distribution_chart({"low": 5, "medium": 3, "high": 2}).data[0].y) == [5, 3, 2]
    bins = [{"start": i / 20, "end": (i + 1) / 20, "count": i} for i in range(20)]
    assert len(charts.churn_histogram_chart(bins).data[0].x) == 20
    assert charts.probability_gauge(0.8).data[0].value == pytest.approx(80.0)


def test_shap_chart_colours_by_direction(charts):
    features = [
        {"feature": "recency_days", "value": 300.0, "shap_value": 0.5, "effect": "increases churn risk"},
        {"feature": "frequency", "value": 9.0, "shap_value": -0.2, "effect": "reduces churn risk"},
    ]
    bars = charts.shap_chart(features).data[0]
    assert list(bars.x) == [-0.2, 0.5]  # sorted so the largest effect is at the top
    assert list(bars.marker.color) == [charts.DOWN_COLOR, charts.UP_COLOR]


def test_metric_comparison_chart(charts):
    frame = pd.DataFrame(
        {"model": ["xgboost", "lstm", "svm"], "family": ["Classical ML", "Deep learning", "Classical ML"],
         "test_roc_auc": [0.82, 0.80, None]}
    )
    figure = charts.metric_comparison_chart(frame, "test_roc_auc", "ROC-AUC", target=0.8)
    assert len(figure.data) == 2  # one trace per model family
    assert sum(len(trace.y) for trace in figure.data) == 2  # the missing value is dropped