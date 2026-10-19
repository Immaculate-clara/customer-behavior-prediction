"""Train and compare Customer Lifetime Value (CLV) regression models.

Target: ``future_revenue`` = sales revenue in the churn-horizon window after the
cutoff (about 57% of customers spend exactly 0 and a few spend very large amounts).

Method
------
* Same stratified train/val/test split as the churn models (``data_prep``).
* Models are trained on ``log1p(revenue)`` and predictions are converted back to
  currency, which keeps a handful of huge wholesale accounts from dominating the fit.
* Hyperparameters listed as ``[a, b, c]`` in config.yaml (random forest, XGBoost,
  LightGBM sections) are tuned by cross-validated random search on TRAIN only,
  scored by R-squared on the log scale. Ridge uses ``models.ridge.alpha`` when present.
* Models are ranked by VALIDATION log-scale R-squared. Raw-currency metrics (MAE,
  RMSE, R-squared) are reported on the test set; the PRD target is raw R-squared.

Run:  python -m src.models.train_clv [--models ridge,random_forest]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from math import prod
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.models.data_prep import CHURN_TARGET, CLV_TARGET, ModelData, make_preprocessor, prepare_data
from src.models.evaluate import log_run, regression_metrics
from src.models.train_classical import _split_config
from src.utils.config import get_config_value, get_seed, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_PREFIX = "clv_"
DEFAULT_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def _log_r2(y_true, y_pred) -> float:
    """R-squared on the log1p scale (robust to a few extreme accounts)."""
    return float(r2_score(np.log1p(np.asarray(y_true)), np.log1p(np.clip(np.asarray(y_pred), 0, None))))


@dataclass
class Candidate:
    name: str
    estimator: BaseEstimator
    scaled: bool
    search_space: dict[str, list] = field(default_factory=dict)


def build_candidates(seed: int) -> dict[str, Candidate]:
    """Create the regression models from the ``models`` section of config.yaml."""
    cfg = get_config_value("models", {})
    candidates: dict[str, Candidate] = {}

    alphas = cfg.get("ridge", {}).get("alpha", DEFAULT_RIDGE_ALPHAS)
    candidates["ridge"] = Candidate(
        "ridge", Ridge(), scaled=True, search_space={"alpha": alphas if isinstance(alphas, list) else [alphas]}
    )

    fixed, space = _split_config(cfg.get("random_forest", {}))
    candidates["random_forest"] = Candidate(
        "random_forest",
        RandomForestRegressor(**fixed, n_jobs=1, random_state=seed),
        scaled=False,
        search_space=space,
    )

    from xgboost import XGBRegressor  # imported lazily so other models work without it

    fixed, space = _split_config(cfg.get("xgboost", {}), exclude=("early_stopping_rounds",))
    candidates["xgboost"] = Candidate(
        "xgboost",
        XGBRegressor(**fixed, objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=seed),
        scaled=False,
        search_space=space,
    )

    from lightgbm import LGBMRegressor

    fixed, space = _split_config(cfg.get("lightgbm", {}), exclude=("early_stopping_rounds",))
    candidates["lightgbm"] = Candidate(
        "lightgbm",
        LGBMRegressor(**fixed, n_jobs=1, random_state=seed, verbose=-1),
        scaled=False,
        search_space=space,
    )
    return candidates


def train_candidate(candidate: Candidate, data: ModelData, seed: int) -> tuple[TransformedTargetRegressor, dict[str, Any]]:
    """Tune and fit one candidate on log1p(revenue); predictions come back in currency."""
    X_train, y_train = data.X("train"), data.y("train", CLV_TARGET)
    pipeline = Pipeline([("prep", make_preprocessor(candidate.scaled)), ("model", candidate.estimator)])
    model = TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False)

    # Folds are stratified on churn so every fold has the same share of zero-revenue customers.
    n_splits = int(get_config_value("training.cv_folds", 5))
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X_train, data.y("train", CHURN_TARGET)))

    scorer = make_scorer(_log_r2)
    if candidate.search_space:
        grid_size = prod(len(values) for values in candidate.search_space.values())
        search = RandomizedSearchCV(
            model,
            {f"regressor__model__{key}": values for key, values in candidate.search_space.items()},
            n_iter=min(int(get_config_value("training.search_iterations", 20)), grid_size),
            scoring=scorer,
            cv=folds,
            n_jobs=-1,
            random_state=seed,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted, cv_score = search.best_estimator_, float(search.best_score_)
        best_params = {key.removeprefix("regressor__model__"): value for key, value in search.best_params_.items()}
    else:
        scores = cross_val_score(model, X_train, y_train, scoring=scorer, cv=folds, n_jobs=-1)
        fitted, cv_score, best_params = model.fit(X_train, y_train), float(scores.mean()), {}

    return fitted, {"cv_r2_log": cv_score, "best_params": best_params}


def _predict(model: TransformedTargetRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predicted revenue in currency, never negative."""
    return np.clip(model.predict(X), 0.0, None)


def evaluate_model(model: TransformedTargetRegressor, data: ModelData) -> dict[str, float]:
    """Validation (log scale, used for ranking) and test metrics (log and raw scale)."""
    y_val, y_test = data.y("val", CLV_TARGET), data.y("test", CLV_TARGET)
    val_pred, test_pred = _predict(model, data.X("val")), _predict(model, data.X("test"))
    raw = regression_metrics(y_test, test_pred)
    return {
        "val_r2_log": _log_r2(y_val, val_pred),
        "test_r2_log": _log_r2(y_test, test_pred),
        "test_r2": raw["r2"],
        "test_mae": raw["mae"],
        "test_rmse": raw["rmse"],
        "test_spearman": float(spearmanr(y_test, test_pred).statistic),
    }


def _baseline_row(data: ModelData) -> dict[str, Any]:
    """Reference row: always predict the average training revenue."""
    y_train, y_val, y_test = (data.y(name, CLV_TARGET) for name in ("train", "val", "test"))
    mean_value = float(y_train.mean())
    raw = regression_metrics(y_test, np.full(len(y_test), mean_value))
    return {
        "model": "baseline_mean",
        "cv_r2_log": np.nan,
        "val_r2_log": _log_r2(y_val, np.full(len(y_val), mean_value)),
        "test_r2_log": _log_r2(y_test, np.full(len(y_test), mean_value)),
        "test_r2": raw["r2"],
        "test_mae": raw["mae"],
        "test_rmse": raw["rmse"],
        "test_spearman": np.nan,
    }


def run(model_names: list[str] | None = None) -> pd.DataFrame:
    """Train every selected regressor, save artifacts and return the comparison table."""
    seed = get_seed()
    data = prepare_data()
    candidates = build_candidates(seed)

    if model_names:
        unknown = sorted(set(model_names) - set(candidates))
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. Available: {sorted(candidates)}")
        candidates = {name: candidates[name] for name in model_names}

    artifacts_dir = resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"), create=True)
    rows = []
    for name, candidate in candidates.items():
        logger.info("Training %s ...", name)
        model, info = train_candidate(candidate, data, seed)
        metrics = evaluate_model(model, data)
        joblib.dump(model, artifacts_dir / f"{ARTIFACT_PREFIX}{name}.joblib")

        row = {"model": name, "cv_r2_log": info["cv_r2_log"], **metrics}
        rows.append(row)
        log_run(
            {
                "task": "clv_regression",
                **row,
                "n_train": len(data.y("train")),
                "seed": seed,
                "params": json.dumps(info["best_params"], default=str),
            }
        )
        logger.info(
            "%s | CV log-R2 %.3f | val log-R2 %.3f | test R2 %.3f | test MAE %.1f",
            name, row["cv_r2_log"], row["val_r2_log"], row["test_r2"], row["test_mae"],
        )

    comparison = pd.DataFrame(rows).sort_values("val_r2_log", ascending=False).reset_index(drop=True)
    comparison.to_csv(artifacts_dir / "clv_model_comparison.csv", index=False)

    metadata = {
        "feature_columns": data.feature_columns,
        "target": CLV_TARGET,
        "target_transform": "log1p (predictions converted back with expm1, clipped at 0)",
        "seed": seed,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_by_val_r2_log": comparison.loc[0, "model"],
    }
    (artifacts_dir / "clv_metadata.json").write_text(json.dumps(metadata, indent=2))

    _report(comparison, _baseline_row(data), float(get_config_value("training.targets.clv_r2", 0.60)))
    return comparison


def _report(comparison: pd.DataFrame, baseline: dict[str, Any], r2_target: float) -> None:
    table = pd.concat([comparison, pd.DataFrame([baseline])], ignore_index=True)
    table["meets_target"] = table["test_r2"] >= r2_target
    print("\nCLV regression comparison (ranked by validation log-scale R2)")
    print(table.round(3).to_string(index=False))
    print(f"\nPRD target: raw test R2 >= {r2_target}")
    print(f"Best model by validation log-scale R2: {comparison.loc[0, 'model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLV regression models")
    parser.add_argument("--models", default="", help="comma-separated subset, e.g. ridge,random_forest")
    args = parser.parse_args()
    run([name.strip() for name in args.models.split(",") if name.strip()] or None)