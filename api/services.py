"""Model loading and prediction logic shared by all API routes.

The service loads every artifact once at start-up and keeps them in memory, so a
request only runs ``predict``. The SHAP explainer is created on first use because
importing it is slow.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_config_value, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "customer_id"


class CustomerNotFound(LookupError):
    """Raised when a customer_id is not in the feature table."""


class InvalidFeatures(ValueError):
    """Raised when a features dictionary is incomplete or contains invalid numbers."""


class PredictionService:
    """Serves churn, CLV, segment and explanation predictions from saved artifacts."""

    def __init__(self) -> None:
        artifacts = resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"))

        # Churn model, its decision threshold and feature list
        self.churn_name = str(get_config_value("api.model_name", "xgboost"))
        self.churn_model = self._load(artifacts / f"churn_{self.churn_name}.joblib", "python -m src.models.train_classical")
        churn_meta = self._load_json(artifacts / "churn_metadata.json", "python -m src.models.train_classical")
        self.feature_columns: list[str] = list(churn_meta["feature_columns"])
        self.threshold = float(churn_meta.get("thresholds", {}).get(self.churn_name, 0.5))

        # CLV model: the best one chosen during training
        clv_meta = self._load_json(artifacts / "clv_metadata.json", "python -m src.models.train_clv")
        self.clv_name = str(clv_meta["best_model_by_val_r2_log"])
        self.clv_model = self._load(artifacts / f"clv_{self.clv_name}.joblib", "python -m src.models.train_clv")

        # Segmentation bundle (preprocessor + K-Means + names) and stored customer segments
        self.segmenter = self._load(artifacts / "segmentation_kmeans.joblib", "python -m src.segmentation.clustering")
        self.segment_names: dict[int, str] = {int(k): v for k, v in self.segmenter["segment_names"].items()}
        segments_path = resolve_path(get_config_value("paths.customer_segments", "data/processed/customer_segments.parquet"))
        self.stored_segments = pd.read_parquet(segments_path).set_index(ID_COLUMN)[["segment_id", "segment_name"]]
        self.segment_profiles = pd.read_csv(artifacts / "segment_profiles.csv")

        # Feature table used for look-ups by customer_id
        table = pd.read_parquet(resolve_path(get_config_value("paths.customer_features", "data/processed/customer_features.parquet")))
        self.features = table.set_index(ID_COLUMN)[self.feature_columns].astype(float)

        low = float(get_config_value("api.risk_thresholds.low", 0.33))
        high = float(get_config_value("api.risk_thresholds.high", 0.66))
        self.risk_low, self.risk_high = low, high

        self._explainer = None
        self._explainer_lock = threading.Lock()
        logger.info("Prediction service ready (churn=%s, clv=%s, customers=%s)", self.churn_name, self.clv_name, f"{len(self.features):,}")

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load(path, fix: str):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact {path}. Create it with: {fix}")
        return joblib.load(path)

    @staticmethod
    def _load_json(path, fix: str) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact {path}. Create it with: {fix}")
        return json.loads(path.read_text())

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def resolve(self, customer_id: int | None, features: dict[str, float] | None) -> tuple[pd.DataFrame, list[int | None]]:
        """Return a one-row feature frame for a known customer or a submitted feature set."""
        if customer_id is not None:
            return self.resolve_many([customer_id])
        assert features is not None
        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            raise InvalidFeatures(f"Missing features: {missing}")
        row = pd.DataFrame([[features[name] for name in self.feature_columns]], columns=self.feature_columns, dtype=float)
        if not np.isfinite(row.to_numpy()).all():
            raise InvalidFeatures("Feature values must be finite numbers")
        return row, [None]

    def resolve_many(self, customer_ids: list[int]) -> tuple[pd.DataFrame, list[int | None]]:
        unknown = [cid for cid in customer_ids if cid not in self.features.index]
        if unknown:
            raise CustomerNotFound(f"Unknown customer_id(s): {unknown[:10]}")
        return self.features.loc[customer_ids].reset_index(drop=True), list(customer_ids)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #
    def risk_level(self, probability: float) -> str:
        if probability < self.risk_low:
            return "low"
        return "medium" if probability < self.risk_high else "high"

    def churn(self, X: pd.DataFrame, ids: list[int | None]) -> list[dict[str, Any]]:
        probabilities = self.churn_model.predict_proba(X[self.feature_columns])[:, 1]
        return [
            {
                "customer_id": cid,
                "churn_probability": float(p),
                "predicted_churn": bool(p >= self.threshold),
                "risk_level": self.risk_level(float(p)),
                "threshold": self.threshold,
                "model": self.churn_name,
            }
            for cid, p in zip(ids, probabilities)
        ]

    def clv(self, X: pd.DataFrame, ids: list[int | None]) -> list[dict[str, Any]]:
        revenue = np.clip(self.clv_model.predict(X[self.feature_columns]), 0.0, None)
        return [{"customer_id": cid, "predicted_revenue": float(r), "model": self.clv_name} for cid, r in zip(ids, revenue)]

    def segment(self, X: pd.DataFrame, ids: list[int | None]) -> list[dict[str, Any]]:
        """Stored segment for known customers, fresh K-Means assignment for new ones."""
        results: list[dict[str, Any]] = [{}] * len(ids)
        new_positions = []
        for position, cid in enumerate(ids):
            if cid is not None and cid in self.stored_segments.index:
                row = self.stored_segments.loc[cid]
                results[position] = {"customer_id": cid, "segment_id": int(row["segment_id"]), "segment_name": str(row["segment_name"])}
            else:
                new_positions.append(position)
        if new_positions:
            features = self.segmenter["features"]
            matrix = self.segmenter["preprocessor"].transform(X.iloc[new_positions][features])
            labels = self.segmenter["kmeans"].predict(np.asarray(matrix, dtype=float))
            for position, label in zip(new_positions, labels):
                results[position] = {
                    "customer_id": ids[position],
                    "segment_id": int(label),
                    "segment_name": self.segment_names[int(label)],
                }
        return results

    def profiles(self, X: pd.DataFrame, ids: list[int | None]) -> list[dict[str, Any]]:
        churn, clv, segment = self.churn(X, ids), self.clv(X, ids), self.segment(X, ids)
        return [{"customer_id": cid, "churn": a, "clv": b, "segment": c} for cid, a, b, c in zip(ids, churn, clv, segment)]

    def explain(self, X: pd.DataFrame, ids: list[int | None], top_n: int) -> list[dict[str, Any]]:
        """SHAP explanation of the churn score (explainer is built on first use)."""
        if self._explainer is None:
            with self._explainer_lock:
                if self._explainer is None:
                    from src.explainability.explain import ChurnExplainer

                    self._explainer = ChurnExplainer(self.churn_name)
        contributions = self._explainer.explain_rows(X[self.feature_columns], top_n=top_n)
        churn = self.churn(X, ids)
        return [
            {"customer_id": cid, "churn_probability": item["churn_probability"], "top_features": top}
            for cid, item, top in zip(ids, churn, contributions)
        ]

    # ------------------------------------------------------------------ #
    # Reference data
    # ------------------------------------------------------------------ #
    def segment_summaries(self) -> list[dict[str, Any]]:
        reverse = {name: segment_id for segment_id, name in self.segment_names.items()}
        summaries = []
        for _, row in self.segment_profiles.iterrows():
            extra = {"customers", "share_pct", "churn_rate", "avg_future_revenue", "segment", "cluster"}
            summaries.append(
                {
                    "segment_id": int(row["cluster"]) if "cluster" in row else int(reverse[row["segment"]]),
                    "segment": str(row["segment"]),
                    "customers": int(row["customers"]),
                    "share_pct": float(row["share_pct"]),
                    "churn_rate": float(row["churn_rate"]) if "churn_rate" in row and pd.notna(row["churn_rate"]) else None,
                    "avg_future_revenue": float(row["avg_future_revenue"]) if "avg_future_revenue" in row and pd.notna(row["avg_future_revenue"]) else None,
                    "averages": {k: float(v) for k, v in row.items() if k not in extra and pd.notna(v)},
                }
            )
        return summaries

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "churn_model": self.churn_name,
            "clv_model": self.clv_name,
            "customers_available": int(len(self.features)),
            "feature_columns": self.feature_columns,
        }