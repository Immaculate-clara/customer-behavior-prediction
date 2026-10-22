"""Tests for the customer segmentation module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.segmentation import clustering
from src.segmentation.clustering import (
    _name_segment,
    choose_k,
    evaluate_kmeans,
    find_outliers,
    prepare_matrix,
    profile_segments,
    run_dbscan,
    run_gmm,
    run_hierarchical,
)

# (recency_days, frequency, monetary_total) of four customer archetypes
ARCHETYPES = {0: (10, 30, 3000), 1: (120, 3, 200), 2: (60, 10, 900), 3: (250, 1, 60)}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    rows = []
    for group in rng.integers(0, 4, n):
        recency, frequency, monetary = ARCHETYPES[int(group)]
        rows.append(
            {
                "recency_days": rng.gamma(4, recency / 4),
                "frequency": rng.poisson(frequency) + 1,
                "monetary_total": rng.lognormal(np.log(monetary), 0.4),
                "monetary_avg": rng.lognormal(4, 0.5),
                "avg_basket_size": rng.lognormal(3, 0.6),
                "tenure_days": rng.uniform(30, 700),
                "interpurchase_mean_days": rng.gamma(3, 20),
                "return_rate": rng.beta(1, 9),
                "discount_line_share": rng.beta(1, 6),
                "seasonal_order_share": rng.uniform(0, 1),
                "purchase_frequency_trend": rng.normal(0, 0.2),
                "category_entropy": rng.uniform(0, 1),
            }
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "customer_id", np.arange(n) + 1000)
    frame["churn"] = (frame["recency_days"] > 80).astype(int)
    frame["future_revenue"] = np.where(frame["churn"] == 1, 0.0, frame["monetary_total"] / 4)
    return frame


@pytest.fixture(scope="module")
def matrix(table):
    return prepare_matrix(table)[0]


# ------------------------------------------------------------------- features
def test_prepare_matrix_returns_a_finite_scaled_matrix(table):
    matrix, features, preprocessor = prepare_matrix(table)
    assert matrix.shape == (len(table), len(features))
    assert np.isfinite(matrix).all()
    assert preprocessor is not None


def test_prepare_matrix_needs_at_least_three_features(table):
    with pytest.raises(ValueError, match="at least 3"):
        prepare_matrix(table[["recency_days", "frequency"]])


def test_prepare_matrix_skips_missing_features(table):
    _, features, _ = prepare_matrix(table.drop(columns=["return_rate"]))
    assert "return_rate" not in features


# ----------------------------------------------------------------- clustering
def test_evaluate_kmeans_scores_every_k(matrix):
    scores = evaluate_kmeans(matrix, range(2, 5), seed=0)
    assert list(scores["k"]) == [2, 3, 4]
    assert {"inertia", "silhouette", "davies_bouldin", "calinski_harabasz"} <= set(scores.columns)
    assert scores["inertia"].is_monotonic_decreasing


def test_choose_k_stays_inside_the_allowed_range():
    scores = pd.DataFrame({"k": range(2, 9), "silhouette": [0.9, 0.5, 0.4, 0.6, 0.3, 0.2, 0.1]})
    assert choose_k(scores, min_clusters=4, max_clusters=6) == 5


def test_choose_k_falls_back_to_all_values_when_none_qualify():
    scores = pd.DataFrame({"k": [2, 3], "silhouette": [0.2, 0.7]})
    assert choose_k(scores, min_clusters=8, max_clusters=10) == 3


def test_gmm_reports_quality_and_agreement(matrix):
    labels = np.random.default_rng(0).integers(0, 4, len(matrix))
    model, metrics = run_gmm(matrix, n_components=4, seed=0, kmeans_labels=labels)
    assert model.n_components == 4
    assert {"silhouette", "bic", "agreement_with_kmeans_ari"} <= set(metrics)


def test_dbscan_reports_noise_share(matrix):
    labels, metrics = run_dbscan(matrix)
    assert len(labels) == len(matrix)
    assert 0.0 <= metrics["noise_share"] <= 1.0
    assert metrics["eps"] > 0


def test_find_outliers_flags_about_five_percent(matrix):
    flags = find_outliers(matrix)
    assert flags.dtype == bool
    assert 0.02 <= flags.mean() <= 0.08


def test_hierarchical_uses_a_sample(matrix):
    labels = np.zeros(len(matrix), dtype=int)
    metrics, sample = run_hierarchical(matrix, n_clusters=3, kmeans_labels=labels, seed=0)
    assert metrics["sample_size"] == len(sample) <= len(matrix)
    assert len(set(sample)) == len(sample)


# --------------------------------------------------------------------- naming
@pytest.mark.parametrize(
    ("scores", "kwargs", "expected"),
    [
        ((0.5, 0.5, 0.5), {"discount_share": 0.5}, "Discount Bulk Buyers"),
        ((0.5, 0.5, 0.5), {"seasonal_share": 0.9}, "Seasonal Shoppers"),
        ((0.9, 0.9, 0.9), {}, "Champions"),
        ((0.6, 0.6, 0.2), {}, "Loyal Customers"),
        ((0.8, 0.1, 0.1), {}, "New & Promising"),
        ((0.1, 0.1, 0.1), {"recency_days": 400}, "Lost Customers"),
        ((0.1, 0.7, 0.1), {}, "At Risk"),
        ((0.1, 0.1, 0.1), {}, "Lapsing Occasional Buyers"),
    ],
)
def test_segment_names_follow_business_rules(scores, kwargs, expected):
    settings = {"recency_days": 100.0, "discount_share": 0.0, "seasonal_share": 0.0, **kwargs}
    assert _name_segment(*scores, **settings) == expected


def test_profiles_have_unique_names_and_full_coverage(table):
    matrix, features, _ = prepare_matrix(table)
    labels = clustering.KMeans(n_clusters=5, n_init=3, random_state=0).fit_predict(matrix)
    profile, names = profile_segments(table, labels, features)

    assert len(set(names.values())) == len(names) == 5
    assert profile["customers"].sum() == len(table)
    assert profile["share_pct"].sum() == pytest.approx(100.0, abs=0.01)
    assert {"churn_rate", "avg_future_revenue", "segment"} <= set(profile.columns)


# ----------------------------------------------------------------- end to end
def _redirect_outputs(monkeypatch, tmp_path: Path) -> None:
    def fake_resolve(relative_path, create=False):
        path = tmp_path / Path(relative_path).name
        if create:
            (path if not path.suffix else path.parent).mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(clustering, "resolve_path", fake_resolve)


def test_run_without_saving_returns_one_segment_per_customer(table, monkeypatch):
    monkeypatch.setattr(clustering, "load_feature_table", lambda: table)
    result = clustering.run(save=False)

    assert len(result) == len(table)
    assert result["customer_id"].is_unique
    assert {"segment_id", "segment_name", "is_outlier", "gmm_segment_id"} <= set(result.columns)
    assert 4 <= result["segment_id"].nunique() <= 6


def test_run_saves_all_artifacts(table, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("matplotlib")
    _redirect_outputs(monkeypatch, tmp_path)
    monkeypatch.setattr(clustering, "load_feature_table", lambda: table)

    clustering.run(save=True)

    expected = [
        "customer_segments.parquet",
        "models_artifacts/segment_profiles.csv",
        "models_artifacts/segmentation_kmeans.joblib",
        "models_artifacts/segmentation_metadata.json",
        "figures/segments_pca.png",
        "figures/kmeans_k_selection.png",
        "figures/hierarchical_dendrogram.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name