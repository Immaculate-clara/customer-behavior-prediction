"""Explain the churn model: SHAP (global and per customer), LIME and partial dependence.

* SHAP  - how much each feature pushes a customer's churn score up or down.
* LIME  - a second, model-agnostic local explanation used as a cross-check.
* PDP   - average effect of selected features (``explainability.pdp_features``).

``ChurnExplainer`` is also what the API uses to explain a single customer.
Explanations are computed on the TEST split; the SHAP background sample comes
from the TRAINING split.

Run:  python -m src.explainability.explain [--model xgboost]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.data_prep import ModelData, prepare_data
from src.utils.config import get_config_value, get_seed, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARTIFACT_PREFIX = "churn_"
BACKGROUND_FILE = "explain_background.joblib"
TREE_MODELS = {"decision_tree", "random_forest", "xgboost", "lightgbm"}


# --------------------------------------------------------------------------- #
# Helpers (no SHAP/LIME needed)
# --------------------------------------------------------------------------- #
def _artifacts_dir():
    return resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"), create=True)


def positive_class_values(values: Any) -> np.ndarray:
    """Return SHAP values for the churn class as an (n_customers, n_features) array."""
    if isinstance(values, list):  # older SHAP versions: one array per class
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, -1]
    return values


def top_contributions(
    shap_row: np.ndarray,
    feature_names: list[str],
    raw_values: np.ndarray,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """The ``top_n`` features with the largest effect on one customer's churn score."""
    order = np.argsort(-np.abs(shap_row))[:top_n]
    return [
        {
            "feature": feature_names[i],
            "value": float(raw_values[i]),
            "shap_value": float(shap_row[i]),
            "effect": "increases churn risk" if shap_row[i] > 0 else "reduces churn risk",
        }
        for i in order
    ]


def load_churn_model(model_name: str):
    path = _artifacts_dir() / f"{ARTIFACT_PREFIX}{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Churn model not found: {path}. Run 'python -m src.models.train_classical' first.")
    return joblib.load(path)


def _feature_names(estimator) -> list[str]:
    metadata_path = _artifacts_dir() / "churn_metadata.json"
    if metadata_path.exists():
        names = json.loads(metadata_path.read_text()).get("feature_columns")
        if names:
            return list(names)
    names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        raise ValueError("Feature names unavailable: train the churn models again to create churn_metadata.json")
    return list(names)


def _apply_prep(prep, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline's preprocessing step and keep the column names."""
    values = X.to_numpy(dtype=float) if isinstance(prep, str) else np.asarray(prep.transform(X), dtype=float)
    return pd.DataFrame(values, columns=X.columns, index=X.index)


# --------------------------------------------------------------------------- #
# SHAP explainer used by the report and by the API
# --------------------------------------------------------------------------- #
class ChurnExplainer:
    """SHAP explanations for one saved churn model."""

    def __init__(self, model_name: str | None = None, data: ModelData | None = None):
        import shap  # imported here so the helpers above work without it

        self.model_name = model_name or str(get_config_value("api.model_name", "xgboost"))
        self.pipeline = load_churn_model(self.model_name)
        self.prep = self.pipeline.named_steps["prep"]
        self.estimator = self.pipeline.named_steps["model"]
        self.feature_names = _feature_names(self.estimator)
        self.background = self._load_background(data)
        self.explainer = self._build_explainer(shap)

    def _load_background(self, data: ModelData | None) -> pd.DataFrame:
        path = _artifacts_dir() / f"{self.model_name}_{BACKGROUND_FILE}"
        if path.exists():
            return joblib.load(path)
        data = data or prepare_data()
        size = int(get_config_value("explainability.shap_background_size", 100))
        train = data.X("train")
        sample = train.sample(n=min(size, len(train)), random_state=get_seed())
        background = _apply_prep(self.prep, sample)
        joblib.dump(background, path)
        return background

    def _build_explainer(self, shap):
        try:
            if self.model_name == "logistic_regression":
                return shap.LinearExplainer(self.estimator, self.background)
            if self.model_name in TREE_MODELS:
                return shap.TreeExplainer(self.estimator)
        except Exception as exc:  # unusual library-version combination: use the slower generic explainer
            logger.warning("Fast SHAP explainer failed (%s); using the generic explainer instead", exc)
        columns = self.background.columns

        def predict(values):
            return self.estimator.predict_proba(pd.DataFrame(values, columns=columns))[:, 1]

        return shap.Explainer(predict, self.background)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values (churn class) for raw feature rows ``X``."""
        explanation = self.explainer(_apply_prep(self.prep, X[self.feature_names]))
        return positive_class_values(getattr(explanation, "values", explanation))

    def explain_rows(self, X: pd.DataFrame, top_n: int = 5) -> list[list[dict[str, Any]]]:
        """Top contributing features for each row of ``X``."""
        X = X[self.feature_names]
        values = self.shap_values(X)
        raw = X.to_numpy(dtype=float)
        return [top_contributions(values[i], self.feature_names, raw[i], top_n) for i in range(len(X))]


# --------------------------------------------------------------------------- #
# LIME and partial dependence (optional extras: a failure is logged, not fatal)
# --------------------------------------------------------------------------- #
def lime_explanations(pipeline, feature_names: list[str], data: ModelData, customers: dict[str, int], seed: int) -> dict[str, Any]:
    from lime.lime_tabular import LimeTabularExplainer

    num_features = int(get_config_value("explainability.lime_num_features", 8))
    train = data.X("train")[feature_names]
    explainer = LimeTabularExplainer(
        train.to_numpy(dtype=float),
        feature_names=feature_names,
        class_names=["retained", "churned"],
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )

    def predict(values: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(pd.DataFrame(values, columns=feature_names))

    test = data.X("test")[feature_names]
    results = {}
    for label, position in customers.items():
        explanation = explainer.explain_instance(
            test.iloc[position].to_numpy(dtype=float), predict, num_features=num_features, num_samples=2000
        )
        results[label] = {
            "customer_id": int(data.ids("test").iloc[position]),
            "weights": [{"rule": rule, "weight": float(weight)} for rule, weight in explanation.as_list()],
        }
    return results


def save_partial_dependence(pipeline, X_train: pd.DataFrame, path) -> list[str]:
    import matplotlib.pyplot as plt
    from sklearn.inspection import PartialDependenceDisplay

    wanted = get_config_value("explainability.pdp_features", ["recency_days", "frequency"])
    features = [name for name in wanted if name in X_train.columns]
    if not features:
        return []
    sample = X_train.sample(n=min(1000, len(X_train)), random_state=get_seed()).astype(float)  # PDP rejects integer columns
    display = PartialDependenceDisplay.from_estimator(pipeline, sample, features, kind="average", grid_resolution=30)
    display.figure_.suptitle("Partial dependence: average predicted churn probability")
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(display.figure_)
    return features


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _save_shap_plots(shap_values: np.ndarray, X_test: pd.DataFrame, feature_names: list[str], figures_dir) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    paths = []
    for kind, filename in (("dot", "shap_summary.png"), ("bar", "shap_importance.png")):
        shap.summary_plot(shap_values, X_test[feature_names], plot_type=kind, show=False, max_display=15)
        figure = plt.gcf()
        figure.tight_layout()
        path = figures_dir / filename
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def run(model_name: str | None = None) -> dict[str, Any]:
    """Create SHAP, LIME and partial-dependence outputs for one churn model."""
    seed = get_seed()
    data = prepare_data()
    explainer = ChurnExplainer(model_name, data)
    names = explainer.feature_names
    artifacts_dir = _artifacts_dir()
    figures_dir = resolve_path(get_config_value("paths.figures_dir", "docs/figures"), create=True)

    X_test = data.X("test")[names]
    logger.info("Computing SHAP values for %s test customers (%s)", f"{len(X_test):,}", explainer.model_name)
    values = explainer.shap_values(X_test)

    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(artifacts_dir / f"{explainer.model_name}_shap_importance.csv", index=False)
    plots = _save_shap_plots(values, X_test, names, figures_dir)

    # Example customers: highest risk, lowest risk and the most typical one.
    proba = explainer.pipeline.predict_proba(X_test)[:, 1]
    order = np.argsort(proba)
    examples = {"highest_risk": int(order[-1]), "lowest_risk": int(order[0]), "typical": int(order[len(order) // 2])}
    raw = X_test.to_numpy(dtype=float)
    shap_examples = {
        label: {
            "customer_id": int(data.ids("test").iloc[position]),
            "churn_probability": float(proba[position]),
            "top_features": top_contributions(values[position], names, raw[position], 5),
        }
        for label, position in examples.items()
    }

    lime_result: dict[str, Any] = {}
    try:
        lime_result = lime_explanations(explainer.pipeline, names, data, examples, seed)
    except Exception as exc:
        logger.warning("LIME step skipped: %s", exc)

    pdp_features: list[str] = []
    try:
        pdp_features = save_partial_dependence(explainer.pipeline, data.X("train")[names], figures_dir / "partial_dependence.png")
        if pdp_features:
            plots.append(str(figures_dir / "partial_dependence.png"))
    except Exception as exc:
        logger.warning("Partial dependence step skipped: %s", exc)

    report = {
        "model": explainer.model_name,
        "explained_customers": len(X_test),
        "top_features": importance.head(10).to_dict(orient="records"),
        "shap_examples": shap_examples,
        "lime_examples": lime_result,
        "pdp_features": pdp_features,
        "plots": plots,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (artifacts_dir / f"{explainer.model_name}_explanations.json").write_text(json.dumps(report, indent=2))
    _print_report(report, importance)
    return report


def _print_report(report: dict[str, Any], importance: pd.DataFrame) -> None:
    print(f"\nExplainability report for the {report['model']} churn model")
    print("\nTop 10 features by mean |SHAP|")
    print(importance.head(10).round(4).to_string(index=False))
    for label, example in report["shap_examples"].items():
        print(f"\nCustomer {example['customer_id']} ({label.replace('_', ' ')}, churn probability {example['churn_probability']:.2f})")
        for item in example["top_features"]:
            print(f"  {item['feature']:<26} value {item['value']:>10.2f}   SHAP {item['shap_value']:+.3f}   {item['effect']}")
    if report["lime_examples"]:
        print("\nLIME cross-check (top rules for the highest-risk customer)")
        for item in report["lime_examples"]["highest_risk"]["weights"][:5]:
            print(f"  {item['rule']:<40} {item['weight']:+.3f}")
    print("\nSaved plots:")
    for path in report["plots"]:
        print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain the churn model")
    parser.add_argument("--model", default=None, help="model name, default: api.model_name from config")
    run(parser.parse_args().model)