"""Tests for the API prediction service, built on small artifacts created in a temp folder."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from api import services
from api.services import CustomerNotFound, InvalidFeatures, PredictionService
from src.models.data_prep import make_preprocessor

COLUMNS = ["recency_days", "frequency", "monetary_total"]
SEGMENT_NAMES = {0: "Alpha", 1: "Beta", 2: "Gamma"}


def fake_resolve(base: Path):
    def resolver(relative_path, create=False):
        path = base / Path(relative_path).name
        if create:
            (path if not path.suffix else path.parent).mkdir(parents=True, exist_ok=True)
        return path

    return resolver


@pytest.fixture()
def service(tmp_path, monkeypatch) -> PredictionService:
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(3)
    n = 600
    table = pd.DataFrame(
        {
            "customer_id": np.arange(n) + 100,
            "recency_days": rng.gamma(3, 50, n),
            "frequency": rng.poisson(4, n) + 1.0,
            "monetary_total": rng.lognormal(6, 1, n),
        }
    )
    table["churn"] = (table["recency_days"] > 150).astype(int)
    table["future_revenue"] = np.where(table["churn"] == 1, 0.0, table["monetary_total"] / 5)

    artifacts = tmp_path / "models_artifacts"
    artifacts.mkdir()

    churn = Pipeline([("prep", "passthrough"), ("model", RandomForestClassifier(20, random_state=0))])
    joblib.dump(churn.fit(table[COLUMNS], table["churn"]), artifacts / "churn_random_forest.joblib")
    (artifacts / "churn_metadata.json").write_text(
        json.dumps({"feature_columns": COLUMNS, "thresholds": {"random_forest": 0.5}})
    )

    regressor = TransformedTargetRegressor(
        Pipeline([("prep", "passthrough"), ("model", RandomForestRegressor(20, random_state=0))]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    joblib.dump(regressor.fit(table[COLUMNS], table["future_revenue"]), artifacts / "clv_random_forest.joblib")
    (artifacts / "clv_metadata.json").write_text(json.dumps({"best_model_by_val_r2_log": "random_forest"}))

    preprocessor = make_preprocessor(True)
    matrix = preprocessor.fit_transform(table[COLUMNS])
    kmeans = KMeans(3, n_init=5, random_state=0).fit(matrix)
    joblib.dump(
        {"preprocessor": preprocessor, "kmeans": kmeans, "features": COLUMNS, "segment_names": SEGMENT_NAMES},
        artifacts / "segmentation_kmeans.joblib",
    )
    pd.DataFrame(
        {
            "customer_id": table["customer_id"],
            "segment_id": kmeans.labels_,
            "segment_name": [SEGMENT_NAMES[i] for i in kmeans.labels_],
        }
    ).to_parquet(tmp_path / "customer_segments.parquet", index=False)
    table.to_parquet(tmp_path / "customer_features.parquet", index=False)
    pd.DataFrame(
        {
            "cluster": [0, 1, 2],
            "segment": ["Alpha", "Beta", "Gamma"],
            "customers": [100, 200, 300],
            "share_pct": [16.7, 33.3, 50.0],
            "recency_days": [10.0, 50.0, 120.0],
            "churn_rate": [0.1, 0.5, 0.9],
            "avg_future_revenue": [90.0, 40.0, 0.0],
        }
    ).to_csv(artifacts / "segment_profiles.csv", index=False)

    config = {
        "api.model_name": "random_forest",
        "paths.artifacts_dir": "models_artifacts",
        "paths.customer_segments": "customer_segments.parquet",
        "paths.customer_features": "customer_features.parquet",
        "api.risk_thresholds.low": 0.33,
        "api.risk_thresholds.high": 0.66,
    }
    monkeypatch.setattr(services, "get_config_value", lambda key, default=None, path=None: config.get(key, default))
    monkeypatch.setattr(services, "resolve_path", fake_resolve(tmp_path))
    return PredictionService()


NEW_CUSTOMER = {"recency_days": 10.0, "frequency": 8.0, "monetary_total": 5000.0}


def test_service_reports_its_state(service):
    health = service.health()
    assert health["status"] == "ok"
    assert health["customers_available"] == 600
    assert health["feature_columns"] == COLUMNS
    assert health["churn_model"] == health["clv_model"] == "random_forest"


def test_known_customer_profile_uses_the_stored_segment(service):
    X, ids = service.resolve(105, None)
    profile = service.profiles(X, ids)[0]

    stored = service.stored_segments.loc[105]
    assert profile["customer_id"] == 105
    assert profile["segment"]["segment_name"] == stored["segment_name"]
    assert 0.0 <= profile["churn"]["churn_probability"] <= 1.0
    assert profile["clv"]["predicted_revenue"] >= 0.0


def test_new_customer_is_assigned_a_segment_by_kmeans(service):
    X, ids = service.resolve(None, NEW_CUSTOMER)
    segment = service.segment(X, ids)[0]

    assert ids == [None] and segment["customer_id"] is None
    assert segment["segment_name"] in SEGMENT_NAMES.values()


def test_predicted_churn_follows_the_threshold(service):
    X, ids = service.resolve_many([100, 101, 102, 103])
    for item in service.churn(X, ids):
        assert item["predicted_churn"] == (item["churn_probability"] >= service.threshold)
        assert item["risk_level"] == service.risk_level(item["churn_probability"])


def test_missing_features_are_rejected(service):
    with pytest.raises(InvalidFeatures, match="Missing features"):
        service.resolve(None, {"recency_days": 1.0})


def test_non_finite_features_are_rejected(service):
    with pytest.raises(InvalidFeatures, match="finite"):
        service.resolve(None, {**NEW_CUSTOMER, "frequency": float("nan")})


def test_unknown_customers_are_reported(service):
    with pytest.raises(CustomerNotFound, match="999999"):
        service.resolve_many([100, 999999])


def test_batch_resolution_keeps_the_order(service):
    X, ids = service.resolve_many([105, 100, 110])
    assert ids == [105, 100, 110] and len(X) == 3
    assert len(service.profiles(X, ids)) == 3


@pytest.mark.parametrize(("probability", "level"), [(0.0, "low"), (0.329, "low"), (0.33, "medium"), (0.659, "medium"), (0.66, "high"), (1.0, "high")])
def test_risk_levels_use_the_configured_thresholds(service, probability, level):
    assert service.risk_level(probability) == level


def test_segment_summaries_include_averages(service):
    summaries = service.segment_summaries()

    assert [s["segment"] for s in summaries] == ["Alpha", "Beta", "Gamma"]
    assert summaries[0]["customers"] == 100 and summaries[0]["churn_rate"] == pytest.approx(0.1)
    assert "recency_days" in summaries[0]["averages"]
    assert "customers" not in summaries[0]["averages"]


def test_segment_summaries_work_without_a_cluster_column(service):
    service.segment_profiles = service.segment_profiles.drop(columns="cluster")
    assert [s["segment_id"] for s in service.segment_summaries()] == [0, 1, 2]


def test_explanations_build_the_explainer_only_once(service, monkeypatch):
    class FakeExplainer:
        created = 0

        def __init__(self, model_name):
            type(self).created += 1

        def explain_rows(self, X, top_n=5):
            return [[{"feature": "recency_days", "value": 1.0, "shap_value": 0.1}][:top_n] for _ in range(len(X))]

    module = types.ModuleType("src.explainability.explain")
    module.ChurnExplainer = FakeExplainer
    monkeypatch.setitem(sys.modules, "src.explainability.explain", module)

    X, ids = service.resolve_many([100, 101])
    first = service.explain(X, ids, top_n=3)
    service.explain(X, ids, top_n=3)

    assert FakeExplainer.created == 1
    assert [item["customer_id"] for item in first] == [100, 101]
    assert first[0]["top_features"][0]["feature"] == "recency_days"


def test_missing_artifacts_explain_how_to_create_them(service, tmp_path):
    (tmp_path / "models_artifacts" / "clv_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="src.models.train_clv"):
        PredictionService()