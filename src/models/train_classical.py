"""Train and compare the classical churn classifiers.

Models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM and
one distance-based model (KNN, or SVM when ``use_svm_instead_of_knn`` is true).

Method
------
* Identical stratified train/val/test split for every model (``data_prep``).
* Hyperparameters listed as ``[a, b, c]`` in config.yaml are tuned with
  cross-validated random search on the TRAIN split only; scalar values are fixed.
* XGBoost / LightGBM are then refitted with early stopping on the validation split.
* The decision threshold is chosen on validation data to reach the target recall.
* Models are ranked by VALIDATION ROC-AUC; test metrics are reported, never used
  to choose anything.

Run:  python -m src.models.train_classical [--models logistic_regression,random_forest]
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from math import prod
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.data_prep import CHURN_TARGET, ModelData, make_preprocessor, prepare_data
from src.models.evaluate import classification_metrics, log_run, select_threshold
from src.utils.config import get_config_value, get_seed, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

BOOSTED_MODELS = {"xgboost", "lightgbm"}
ARTIFACT_PREFIX = "churn_"


@dataclass
class Candidate:
    """A model family plus its fixed settings and hyperparameter search space."""

    name: str
    estimator: BaseEstimator
    scaled: bool
    search_space: dict[str, list] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Candidate construction
# --------------------------------------------------------------------------- #
def _split_config(section: dict, exclude: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """Split a config section into fixed values (scalars) and search space (lists)."""
    fixed, space = {}, {}
    for key, value in section.items():
        if key in exclude:
            continue
        (space if isinstance(value, list) else fixed)[key] = value
    return fixed, space


def _check_imbalance_method() -> None:
    method = get_config_value("training.imbalance.method", "class_weight")
    if method != "class_weight":
        raise ValueError(
            f"training.imbalance.method='{method}' is not supported by this trainer yet; use 'class_weight'."
        )


def build_candidates(y_train: pd.Series, seed: int) -> dict[str, Candidate]:
    """Create every classical model from the ``models`` section of config.yaml."""
    _check_imbalance_method()
    cfg = get_config_value("models", {})
    positives = float(y_train.sum())
    scale_pos_weight = (len(y_train) - positives) / max(positives, 1.0)

    candidates: dict[str, Candidate] = {}

    # Logistic regression ('penalty' is left at its L2 default; the argument is
    # deprecated in recent scikit-learn versions).
    fixed, space = _split_config(cfg.get("logistic_regression", {}), exclude=("penalty",))
    candidates["logistic_regression"] = Candidate(
        "logistic_regression",
        LogisticRegression(**fixed, class_weight="balanced", random_state=seed),
        scaled=True,
        search_space=space,
    )

    fixed, space = _split_config(cfg.get("decision_tree", {}))
    candidates["decision_tree"] = Candidate(
        "decision_tree",
        DecisionTreeClassifier(**fixed, class_weight="balanced", random_state=seed),
        scaled=False,
        search_space=space,
    )

    fixed, space = _split_config(cfg.get("random_forest", {}))
    candidates["random_forest"] = Candidate(
        "random_forest",
        RandomForestClassifier(**fixed, class_weight="balanced", n_jobs=1, random_state=seed),
        scaled=False,
        search_space=space,
    )

    from xgboost import XGBClassifier  # imported lazily so other models work without it

    fixed, space = _split_config(cfg.get("xgboost", {}), exclude=("early_stopping_rounds",))
    candidates["xgboost"] = Candidate(
        "xgboost",
        XGBClassifier(
            **fixed,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
            random_state=seed,
        ),
        scaled=False,
        search_space=space,
    )

    from lightgbm import LGBMClassifier

    fixed, space = _split_config(cfg.get("lightgbm", {}), exclude=("early_stopping_rounds",))
    candidates["lightgbm"] = Candidate(
        "lightgbm",
        LGBMClassifier(**fixed, class_weight="balanced", n_jobs=1, random_state=seed, verbose=-1),
        scaled=False,
        search_space=space,
    )

    # Exactly one distance-based comparison model (PRD: KNN or SVM).
    if cfg.get("use_svm_instead_of_knn", False):
        fixed, space = _split_config(cfg.get("svm", {}))
        candidates["svm"] = Candidate(
            "svm",
            SVC(**fixed, probability=True, class_weight="balanced", random_state=seed),
            scaled=True,
            search_space=space,
        )
    else:
        fixed, space = _split_config(cfg.get("knn", {}))
        candidates["knn"] = Candidate(
            "knn", KNeighborsClassifier(**fixed), scaled=True, search_space=space
        )

    return candidates


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _fit_boosted_with_early_stopping(
    name: str, estimator: BaseEstimator, X_train, y_train, X_val, y_val
) -> BaseEstimator:
    """Refit a boosted model so that the validation split decides the tree count."""
    rounds = int(get_config_value(f"models.{name}.early_stopping_rounds", 30))
    if name == "xgboost":
        estimator.set_params(early_stopping_rounds=rounds)
        estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        import lightgbm as lgb

        with warnings.catch_warnings():  # newer LightGBM versions warn about 'eval_set' but still accept it
            warnings.filterwarnings("ignore", message=".*eval_set.*")
            estimator.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(rounds, verbose=False)],
            )
    return estimator


def train_candidate(candidate: Candidate, data: ModelData, seed: int) -> tuple[Pipeline, dict[str, Any]]:
    """Tune (if needed) and fit one candidate; return the fitted pipeline and its CV info."""
    X_train, y_train = data.X("train"), data.y("train")
    pipeline = Pipeline([("prep", make_preprocessor(candidate.scaled)), ("model", candidate.estimator)])
    folds = StratifiedKFold(n_splits=int(get_config_value("training.cv_folds", 5)), shuffle=True, random_state=seed)

    if candidate.search_space:
        grid_size = prod(len(values) for values in candidate.search_space.values())
        n_iter = min(int(get_config_value("training.search_iterations", 20)), grid_size)
        search = RandomizedSearchCV(
            pipeline,
            {f"model__{key}": values for key, values in candidate.search_space.items()},
            n_iter=n_iter,
            scoring="roc_auc",
            cv=folds,
            n_jobs=-1,
            random_state=seed,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
        cv_auc, cv_std = float(search.best_score_), float(search.cv_results_["std_test_score"][search.best_index_])
        fitted = search.best_estimator_
    else:
        scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=folds, n_jobs=-1)
        best_params = {}
        cv_auc, cv_std = float(scores.mean()), float(scores.std())
        fitted = pipeline.fit(X_train, y_train)

    if candidate.name in BOOSTED_MODELS:
        tuned = clone(candidate.estimator).set_params(**best_params)
        tuned = _fit_boosted_with_early_stopping(
            candidate.name, tuned, X_train, y_train, data.X("val"), data.y("val")
        )
        fitted = Pipeline([("prep", "passthrough"), ("model", tuned)])

    return fitted, {"cv_roc_auc": cv_auc, "cv_roc_auc_std": cv_std, "best_params": best_params}


def evaluate_pipeline(pipeline: Pipeline, data: ModelData, min_recall: float) -> dict[str, Any]:
    """Pick the threshold on validation data, then score validation and test."""
    val_proba = pipeline.predict_proba(data.X("val"))[:, 1]
    test_proba = pipeline.predict_proba(data.X("test"))[:, 1]
    threshold = select_threshold(data.y("val"), val_proba, min_recall=min_recall)
    return {
        "threshold": threshold,
        "val": classification_metrics(data.y("val"), val_proba, threshold),
        "test": classification_metrics(data.y("test"), test_proba, threshold),
    }


def run(model_names: list[str] | None = None) -> pd.DataFrame:
    """Train every selected model, save artifacts and return the comparison table."""
    seed = get_seed()
    data = prepare_data()
    candidates = build_candidates(data.y("train"), seed)

    if model_names:
        unknown = sorted(set(model_names) - set(candidates))
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. Available: {sorted(candidates)}")
        candidates = {name: candidates[name] for name in model_names}

    targets = get_config_value("training.targets", {})
    min_recall = float(targets.get("recall", 0.70))
    artifacts_dir = resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"), create=True)

    rows, thresholds = [], {}
    for name, candidate in candidates.items():
        logger.info("Training %s ...", name)
        pipeline, cv_info = train_candidate(candidate, data, seed)
        scores = evaluate_pipeline(pipeline, data, min_recall)
        joblib.dump(pipeline, artifacts_dir / f"{ARTIFACT_PREFIX}{name}.joblib")
        thresholds[name] = scores["threshold"]

        row = {
            "model": name,
            "cv_roc_auc": cv_info["cv_roc_auc"],
            "val_roc_auc": scores["val"]["roc_auc"],
            "test_roc_auc": scores["test"]["roc_auc"],
            "test_recall": scores["test"]["recall"],
            "test_precision": scores["test"]["precision"],
            "test_f1": scores["test"]["f1"],
            "test_accuracy": scores["test"]["accuracy"],
            "threshold": scores["threshold"],
        }
        rows.append(row)
        log_run(
            {
                "task": "churn_classification",
                **row,
                "n_train": len(data.y("train")),
                "seed": seed,
                "params": json.dumps(cv_info["best_params"], default=str),
            }
        )
        logger.info(
            "%s | CV AUC %.3f | val AUC %.3f | test AUC %.3f | test recall %.3f",
            name, row["cv_roc_auc"], row["val_roc_auc"], row["test_roc_auc"], row["test_recall"],
        )

    comparison = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(artifacts_dir / "churn_model_comparison.csv", index=False)

    metadata_path = artifacts_dir / "churn_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"thresholds": {}}
    metadata.update(
        {
            "feature_columns": data.feature_columns,
            "target": CHURN_TARGET,
            "seed": seed,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "best_model_by_val_auc": comparison.loc[0, "model"],
        }
    )
    metadata["thresholds"].update(thresholds)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    _report(comparison, targets)
    return comparison


def _report(comparison: pd.DataFrame, targets: dict) -> None:
    auc_target, recall_target = float(targets.get("roc_auc", 0.80)), float(targets.get("recall", 0.70))
    table = comparison.copy()
    table["meets_targets"] = (table["test_roc_auc"] >= auc_target) & (table["test_recall"] >= recall_target)
    print("\nChurn model comparison (ranked by validation ROC-AUC)")
    print(table.round(3).to_string(index=False))
    print(f"\nTargets: ROC-AUC >= {auc_target}, recall >= {recall_target}")
    print(f"Best model by validation ROC-AUC: {comparison.loc[0, 'model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classical churn models")
    parser.add_argument("--models", default="", help="comma-separated subset, e.g. logistic_regression,random_forest")
    args = parser.parse_args()
    run([name.strip() for name in args.models.split(",") if name.strip()] or None)