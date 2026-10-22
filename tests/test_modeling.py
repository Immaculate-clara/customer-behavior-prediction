"""Tests for model data preparation (splits, preprocessing) and evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import data_prep, evaluate
from src.models.data_prep import (
    SPLIT_NAMES,
    SignedLog1p,
    assign_splits,
    get_feature_columns,
    load_feature_table,
    make_preprocessor,
    prepare_data,
)


@pytest.fixture()
def table() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 1200
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(n) + 12000,
            "frequency": rng.poisson(5, n) + 1,
            "monetary_total": rng.lognormal(7, 1.2, n),
            "recency_days": rng.uniform(0, 300, n),
            "top_category": rng.choice(["a", "b"], n),
            "churn": (rng.random(n) < 0.567).astype(int),
        }
    )
    frame["future_revenue"] = np.where(frame["churn"] == 1, 0.0, rng.lognormal(5, 1, n))
    return frame


# ---------------------------------------------------------------- preprocessing
def test_signed_log1p_transforms_only_skewed_columns():
    rng = np.random.default_rng(0)
    skewed = rng.lognormal(3, 1.5, 500)
    symmetric = rng.normal(0, 1, 500)
    X = np.column_stack([skewed, symmetric])

    transformer = SignedLog1p().fit(X)
    result = transformer.transform(X)

    assert list(transformer.skewed_idx_) == [0]
    np.testing.assert_allclose(result[:, 1], symmetric)
    np.testing.assert_allclose(result[:, 0], np.log1p(skewed))


def test_signed_log1p_keeps_the_sign_of_negative_values():
    transformer = SignedLog1p().fit(np.array([[1.0], [2.0], [100.0], [5000.0]]))
    result = transformer.transform(np.array([[-100.0], [0.0], [100.0]])).ravel()
    assert result[0] == pytest.approx(-result[2])
    assert result[1] == 0.0


def test_scaled_preprocessor_standardises_the_training_data(table):
    X = table[["frequency", "monetary_total", "recency_days"]]
    scaled = make_preprocessor(scaled=True).fit_transform(X)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-8)


def test_tree_models_use_passthrough():
    assert make_preprocessor(scaled=False) == "passthrough"


# ------------------------------------------------------------------- splitting
def test_feature_columns_exclude_targets_ids_and_text(table):
    columns = get_feature_columns(table)
    assert columns == ["frequency", "monetary_total", "recency_days"]


def test_split_is_70_15_15_stratified_and_disjoint(table):
    data = prepare_data(table)
    sizes = {name: len(data.y(name)) for name in SPLIT_NAMES}
    assert sizes == {"train": 840, "val": 180, "test": 180}

    rates = [data.y(name).mean() for name in SPLIT_NAMES]
    assert max(rates) - min(rates) < 0.01

    ids = [set(data.ids(name)) for name in SPLIT_NAMES]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_is_deterministic(table):
    pd.testing.assert_series_equal(assign_splits(table), assign_splits(table))


def test_model_data_accessors_return_matching_rows(table):
    data = prepare_data(table)
    assert list(data.X("train").columns) == data.feature_columns
    assert len(data.X("val")) == len(data.y("val")) == len(data.ids("val"))


@pytest.mark.parametrize("missing", ["customer_id", "churn", "future_revenue"])
def test_prepare_data_requires_key_columns(table, missing):
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_data(table.drop(columns=missing))


def test_prepare_data_rejects_missing_values(table):
    table.loc[3, "frequency"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        prepare_data(table)


def test_prepare_data_needs_numeric_features(table):
    with pytest.raises(ValueError, match="No numeric feature columns"):
        prepare_data(table[["customer_id", "churn", "future_revenue"]])


def test_load_feature_table_reads_the_parquet_file(table, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    path = tmp_path / "features.parquet"
    monkeypatch.setattr(data_prep, "resolve_path", lambda _: path)

    with pytest.raises(FileNotFoundError, match="build_features"):
        load_feature_table()

    table.to_parquet(path)
    assert len(load_feature_table()) == len(table)


# ------------------------------------------------------------------ evaluation
def test_classification_metrics_for_a_perfect_model():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate.classification_metrics(y, [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 0, 0, 2)


def test_classification_metrics_respect_the_threshold():
    y = np.array([0, 0, 1, 1])
    proba = [0.1, 0.4, 0.45, 0.9]
    low = evaluate.classification_metrics(y, proba, threshold=0.3)
    high = evaluate.classification_metrics(y, proba, threshold=0.6)
    assert low["recall"] == 1.0 and low["fp"] == 1
    assert high["recall"] == 0.5 and high["fp"] == 0


def test_select_threshold_meets_the_recall_target():
    rng = np.random.default_rng(1)
    y = np.array([0] * 200 + [1] * 200)
    proba = np.concatenate([rng.uniform(0, 0.7, 200), rng.uniform(0.3, 1.0, 200)])

    threshold = evaluate.select_threshold(y, proba, min_recall=0.9)
    assert (proba[y == 1] >= threshold).mean() >= 0.9


def test_select_threshold_falls_back_when_the_target_is_unreachable():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.3, 0.6, 0.9])
    assert evaluate.select_threshold(y, proba, min_recall=1.5) == evaluate.select_threshold(y, proba)


def test_regression_metrics_for_exact_predictions():
    metrics = evaluate.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_log_run_appends_a_row_per_call(tmp_path, monkeypatch):
    path = tmp_path / "model_runs.csv"
    monkeypatch.setattr(evaluate, "resolve_path", lambda *_args, **_kwargs: path)

    evaluate.log_run({"model": "xgboost", "roc_auc": 0.9})
    evaluate.log_run({"model": "logistic", "roc_auc": 0.8})

    log = pd.read_csv(path)
    assert list(log["model"]) == ["xgboost", "logistic"]
    assert "timestamp" in log.columns