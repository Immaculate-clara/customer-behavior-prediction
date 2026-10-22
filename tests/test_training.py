"""Tests for the churn and CLV training scripts, using tiny synthetic data."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from src.models import evaluate, train_classical, train_clv
from src.models.data_prep import prepare_data


def fake_resolve(base: Path):
    def resolver(relative_path, create=False):
        path = base / Path(relative_path).name
        if create:
            (path if not path.suffix else path.parent).mkdir(parents=True, exist_ok=True)
        return path

    return resolver


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    n = 600
    table = pd.DataFrame(
        {
            "customer_id": np.arange(n) + 12000,
            "frequency": rng.poisson(4, n) + 1,
            "monetary_total": rng.lognormal(7, 1.2, n),
            "recency_days": rng.uniform(0, 400, n),
            "tenure_days": rng.uniform(0, 600, n),
        }
    )
    logit = 0.012 * (table["recency_days"] - 200) - 0.25 * (table["frequency"] - 4) + rng.normal(0, 1, n)
    table["churn"] = (logit > np.quantile(logit, 0.43)).astype(int)
    spend = np.exp(0.5 * np.log(table["monetary_total"]) + 0.2 * table["frequency"] + rng.normal(0, 0.8, n)) / 10
    table["future_revenue"] = np.where(table["churn"] == 1, 0.0, spend)
    return prepare_data(table)


@pytest.fixture()
def outputs(tmp_path, monkeypatch, data):
    """Send every file written by the trainers to a temporary folder."""
    for module in (train_classical, train_clv, evaluate):
        monkeypatch.setattr(module, "resolve_path", fake_resolve(tmp_path))
    monkeypatch.setattr(train_classical, "prepare_data", lambda: data)
    monkeypatch.setattr(train_clv, "prepare_data", lambda: data)
    return tmp_path


# =================================================================== churn models
def test_split_config_separates_fixed_values_from_search_spaces():
    fixed, space = train_classical._split_config({"C": 1.0, "depth": [3, 5], "skip": 1}, exclude=("skip",))
    assert fixed == {"C": 1.0} and space == {"depth": [3, 5]}


def test_unsupported_imbalance_method_is_rejected(monkeypatch):
    monkeypatch.setattr(train_classical, "get_config_value", lambda key, default=None, path=None: "smote")
    with pytest.raises(ValueError, match="not supported"):
        train_classical._check_imbalance_method()


def test_all_expected_churn_candidates_are_built(data):
    candidates = train_classical.build_candidates(data.y("train"), seed=0)
    assert {"logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"} <= set(candidates)
    assert ("knn" in candidates) != ("svm" in candidates)  # exactly one distance-based model


def test_svm_can_replace_knn(data, monkeypatch):
    real = train_classical.get_config_value

    def config(key, default=None, path=None):
        if key == "models":
            return {**real("models", {}), "use_svm_instead_of_knn": True}
        return real(key, default, path)

    monkeypatch.setattr(train_classical, "get_config_value", config)
    candidates = train_classical.build_candidates(data.y("train"), seed=0)
    assert "svm" in candidates and "knn" not in candidates


def test_candidate_without_search_space_is_cross_validated(data):
    candidate = train_classical.Candidate("lr", LogisticRegression(max_iter=500), scaled=True)
    pipeline, info = train_classical.train_candidate(candidate, data, seed=0)

    assert info["best_params"] == {} and 0.5 < info["cv_roc_auc"] <= 1.0
    assert pipeline.predict_proba(data.X("val")).shape == (len(data.X("val")), 2)


def test_candidate_with_search_space_is_tuned(data):
    candidate = train_classical.Candidate(
        "lr", LogisticRegression(max_iter=500), scaled=True, search_space={"C": [0.1, 1.0]}
    )
    _, info = train_classical.train_candidate(candidate, data, seed=0)
    assert info["best_params"]["C"] in (0.1, 1.0)


def test_pipeline_evaluation_reports_validation_and_test(data):
    candidate = train_classical.Candidate("lr", LogisticRegression(max_iter=500), scaled=True)
    pipeline, _ = train_classical.train_candidate(candidate, data, seed=0)

    scores = train_classical.evaluate_pipeline(pipeline, data, min_recall=0.7)

    assert 0.0 < scores["threshold"] < 1.0
    assert scores["val"]["recall"] >= 0.7
    assert {"roc_auc", "precision", "f1"} <= set(scores["test"])


def test_boosted_models_use_early_stopping(data):
    pytest.importorskip("xgboost.sklearn")
    pytest.importorskip("lightgbm.sklearn")
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    args = (data.X("train"), data.y("train"), data.X("val"), data.y("val"))
    xgb = train_classical._fit_boosted_with_early_stopping(
        "xgboost", XGBClassifier(n_estimators=60, eval_metric="auc", tree_method="hist"), *args
    )
    lgbm = train_classical._fit_boosted_with_early_stopping("lightgbm", LGBMClassifier(n_estimators=60, verbose=-1), *args)

    assert xgb.predict_proba(args[2]).shape[1] == 2
    assert lgbm.predict_proba(args[2]).shape[1] == 2


def test_churn_run_saves_models_comparison_and_thresholds(outputs):
    result = train_classical.run(["logistic_regression", "decision_tree"])

    assert set(result["model"]) == {"logistic_regression", "decision_tree"}
    assert (outputs / "models_artifacts" / "churn_logistic_regression.joblib").exists()
    assert (outputs / "models_artifacts" / "churn_model_comparison.csv").exists()
    metadata = json.loads((outputs / "models_artifacts" / "churn_metadata.json").read_text())
    assert set(metadata["thresholds"]) == {"logistic_regression", "decision_tree"}
    assert metadata["best_model_by_val_auc"] == result.loc[0, "model"]
    assert len(pd.read_csv(outputs / "model_runs.csv")) == 2


def test_later_churn_runs_keep_earlier_thresholds(outputs):
    train_classical.run(["logistic_regression"])
    train_classical.run(["decision_tree"])
    metadata = json.loads((outputs / "models_artifacts" / "churn_metadata.json").read_text())
    assert set(metadata["thresholds"]) == {"logistic_regression", "decision_tree"}


def test_unknown_churn_model_is_rejected(outputs):
    with pytest.raises(ValueError, match="Unknown model"):
        train_classical.run(["no_such_model"])


# ==================================================================== CLV models
def test_log_r2_is_one_for_exact_predictions():
    assert train_clv._log_r2([0.0, 10.0, 500.0], [0.0, 10.0, 500.0]) == pytest.approx(1.0)


def test_log_r2_ignores_negative_predictions():
    assert np.isfinite(train_clv._log_r2([0.0, 10.0, 500.0], [-5.0, 10.0, 500.0]))


def test_all_expected_clv_candidates_are_built():
    candidates = train_clv.build_candidates(seed=0)
    assert set(candidates) == {"ridge", "random_forest", "xgboost", "lightgbm"}


def test_clv_candidate_is_trained_on_the_log_scale(data):
    candidate = train_clv.Candidate("ridge", Ridge(), scaled=True, search_space={"alpha": [1.0, 10.0]})
    model, info = train_clv.train_candidate(candidate, data, seed=0)

    assert info["best_params"]["alpha"] in (1.0, 10.0)
    assert (train_clv._predict(model, data.X("val")) >= 0).all()


def test_clv_candidate_without_search_space(data):
    candidate = train_clv.Candidate("ridge", Ridge(), scaled=True)
    _, info = train_clv.train_candidate(candidate, data, seed=0)
    assert info["best_params"] == {} and np.isfinite(info["cv_r2_log"])


def test_clv_evaluation_and_baseline(data):
    candidate = train_clv.Candidate("ridge", Ridge(), scaled=True)
    model, _ = train_clv.train_candidate(candidate, data, seed=0)

    metrics = train_clv.evaluate_model(model, data)
    baseline = train_clv._baseline_row(data)

    assert {"val_r2_log", "test_r2", "test_mae", "test_rmse", "test_spearman"} <= set(metrics)
    assert baseline["model"] == "baseline_mean" and baseline["test_r2"] <= 0.0
    assert metrics["val_r2_log"] > baseline["val_r2_log"]


def test_clv_run_saves_models_and_names_the_best(outputs):
    result = train_clv.run(["ridge"])

    assert (outputs / "models_artifacts" / "clv_ridge.joblib").exists()
    assert (outputs / "models_artifacts" / "clv_model_comparison.csv").exists()
    metadata = json.loads((outputs / "models_artifacts" / "clv_metadata.json").read_text())
    assert metadata["best_model_by_val_r2_log"] == result.loc[0, "model"] == "ridge"
    assert joblib.load(outputs / "models_artifacts" / "clv_ridge.joblib") is not None


def test_unknown_clv_model_is_rejected(outputs):
    with pytest.raises(ValueError, match="Unknown model"):
        train_clv.run(["no_such_model"])