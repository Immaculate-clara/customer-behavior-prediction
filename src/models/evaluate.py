"""Evaluation helpers: metrics, decision-threshold selection and run logging."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import get_config_value, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix.

    The positive class is ``1`` (churned customer); ``recall`` is therefore the
    share of churners the model catches.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true, proba, min_recall: float | None = None) -> float:
    """Pick a decision threshold on validation data.

    Chooses the threshold with the highest balanced accuracy (the mean of recall
    and specificity) among those reaching ``min_recall``. Balanced accuracy is used
    instead of F1 because churn is the majority class, where F1 rewards flagging
    almost every customer as a churner. If no threshold reaches ``min_recall``,
    the best balanced-accuracy threshold overall is returned.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(proba))
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    score = (tpr + (1.0 - fpr)) / 2.0

    if min_recall is not None:
        eligible = tpr >= min_recall
        if eligible.any():
            score = np.where(eligible, score, -1.0)
    return float(thresholds[int(np.argmax(score))])

def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R-squared (used for the CLV regression task)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_run(record: dict) -> None:
    """Append one experiment record to the run log CSV (``paths.run_log``)."""
    path = resolve_path(get_config_value("paths.run_log", "logs/model_runs.csv"), create=True)
    row = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), **record}])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)