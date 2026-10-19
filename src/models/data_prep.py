"""Model-ready data: feature selection, stratified 70/15/15 split and preprocessing.

Every training script uses ``prepare_data`` so that all models (classical,
deep learning, CLV regression) are trained and compared on identical splits.
Preprocessing transformers are fitted on the training split only.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config_value, get_seed, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "customer_id"
CHURN_TARGET = "churn"
CLV_TARGET = "future_revenue"
NEXT_CATEGORY_TARGET = "next_category"

# Targets, identifiers and text columns are never model inputs.
NON_FEATURE_COLUMNS = {ID_COLUMN, CHURN_TARGET, CLV_TARGET, NEXT_CATEGORY_TARGET, "top_category"}
SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
class SignedLog1p(BaseEstimator, TransformerMixin):
    """Apply ``sign(x) * log1p(|x|)`` to strongly skewed columns only.

    Which columns are skewed (``|skewness| > skew_threshold``) is decided on the
    data passed to ``fit`` (the training split), then reused for every transform.
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = np.nan_to_num(skew(values, axis=0))
        self.n_features_in_ = values.shape[1]
        self.skewed_idx_ = np.flatnonzero(np.abs(skewness) > self.skew_threshold)
        return self

    def transform(self, X):
        values = np.array(X, dtype=float, copy=True)
        cols = self.skewed_idx_
        values[:, cols] = np.sign(values[:, cols]) * np.log1p(np.abs(values[:, cols]))
        return values


def make_preprocessor(scaled: bool = True):
    """Return the preprocessing step for a model family.

    ``scaled=True``  -> skew-aware log transform + StandardScaler (logistic
    regression, KNN, SVM, neural networks).
    ``scaled=False`` -> unchanged features (tree-based models, per the PRD).
    """
    if not scaled:
        return "passthrough"
    return Pipeline([("log", SignedLog1p()), ("scale", StandardScaler())])


# --------------------------------------------------------------------------- #
# Loading and splitting
# --------------------------------------------------------------------------- #
def load_feature_table() -> pd.DataFrame:
    """Read the customer-level feature table written by the feature builder."""
    path = resolve_path(
        get_config_value("paths.customer_features", "data/processed/customer_features.parquet")
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {path}. Run 'python -m src.features.build_features' first."
        )
    return pd.read_parquet(path).reset_index(drop=True)


def get_feature_columns(table: pd.DataFrame) -> list[str]:
    """Numeric model inputs: every numeric column that is not a target or identifier."""
    numeric = table.select_dtypes(include="number").columns
    return [col for col in numeric if col not in NON_FEATURE_COLUMNS]


def assign_splits(table: pd.DataFrame) -> pd.Series:
    """Assign each customer to train / val / test (default 70 / 15 / 15).

    The split is stratified on churn and deterministic for a given seed, so every
    model and every task (churn, CLV) sees exactly the same customers.
    """
    test_size = float(get_config_value("training.test_size", 0.15))
    val_size = float(get_config_value("training.val_size", 0.15))
    seed = get_seed()

    train_val_idx, test_idx = train_test_split(
        table.index, test_size=test_size, stratify=table[CHURN_TARGET], random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1.0 - test_size),
        stratify=table.loc[train_val_idx, CHURN_TARGET],
        random_state=seed,
    )

    split = pd.Series("train", index=table.index, name="split", dtype="object")
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"
    return split


@dataclass
class ModelData:
    """Feature table, split assignment and feature list with convenient accessors."""

    table: pd.DataFrame
    split: pd.Series
    feature_columns: list[str]

    def X(self, name: str) -> pd.DataFrame:
        return self.table.loc[self.split == name, self.feature_columns]

    def y(self, name: str, target: str = CHURN_TARGET) -> pd.Series:
        return self.table.loc[self.split == name, target]

    def ids(self, name: str) -> pd.Series:
        return self.table.loc[self.split == name, ID_COLUMN]


def prepare_data(table: pd.DataFrame | None = None) -> ModelData:
    """Load the feature table, validate it and attach the train/val/test split."""
    table = load_feature_table() if table is None else table.reset_index(drop=True)

    missing = [col for col in (ID_COLUMN, CHURN_TARGET, CLV_TARGET) if col not in table.columns]
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    feature_columns = get_feature_columns(table)
    if not feature_columns:
        raise ValueError("No numeric feature columns found in the feature table")

    values = table[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c in feature_columns if not np.isfinite(table[c].to_numpy(dtype=float)).all()]
        raise ValueError(f"Features contain NaN or infinite values: {bad}")

    data = ModelData(table=table, split=assign_splits(table), feature_columns=feature_columns)

    for name in SPLIT_NAMES:
        y = data.y(name)
        logger.info("%-5s %6s customers | churn rate %.1f%%", name, f"{len(y):,}", y.mean() * 100)
    logger.info("Using %s features", len(feature_columns))
    return data


if __name__ == "__main__":
    prepared = prepare_data()
    print(prepared.feature_columns)