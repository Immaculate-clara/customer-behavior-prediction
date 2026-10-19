"""Customer segmentation: K-Means (primary) plus GMM, DBSCAN and hierarchical checks.

Steps
-----
1. Select behavioural features (recency, frequency, spend, basket, timing, returns ...),
   apply the same skew-aware scaling used by the other models and fit it on all customers.
2. K-Means for every k in ``segmentation.kmeans.k_range``; choose k by silhouette score
   (never fewer than ``segmentation.min_clusters`` so the result stays useful for marketing).
3. Compare with a second algorithm (GMM by default), DBSCAN (outlier customers) and
   Ward hierarchical clustering on a random sample.
4. Profile every cluster, give it a business name (Champions, Loyal Customers, ...)
   and save labels, profiles, plots and a reusable model bundle.

Run:  python -m src.models.segmentation
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from src.models.data_prep import CHURN_TARGET, CLV_TARGET, ID_COLUMN, load_feature_table, make_preprocessor
from src.utils.config import get_config_value, get_seed, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg",
    "avg_basket_size",
    "tenure_days",
    "interpurchase_mean_days",
    "return_rate",
    "discount_line_share",
    "seasonal_order_share",
    "purchase_frequency_trend",
    "category_entropy",
]
DEFAULT_K_RANGE = [2, 10]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _select_features(table: pd.DataFrame) -> list[str]:
    wanted = get_config_value("segmentation.features", DEFAULT_FEATURES)
    present = [col for col in wanted if col in table.columns]
    missing = sorted(set(wanted) - set(present))
    if missing:
        logger.warning("Segmentation features not in the table (skipped): %s", missing)
    if len(present) < 3:
        raise ValueError(f"Need at least 3 segmentation features, found {present}")
    return present


def prepare_matrix(table: pd.DataFrame) -> tuple[np.ndarray, list[str], Any]:
    """Return the scaled feature matrix, the feature names and the fitted preprocessor."""
    features = _select_features(table)
    preprocessor = make_preprocessor(scaled=True)
    matrix = np.asarray(preprocessor.fit_transform(table[features]), dtype=float)
    return matrix, features, preprocessor


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def _cluster_quality(matrix: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    """Silhouette (higher is better), Davies-Bouldin (lower), Calinski-Harabasz (higher)."""
    return {
        "silhouette": float(silhouette_score(matrix, labels, sample_size=min(len(matrix), 5000), random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
    }


def evaluate_kmeans(matrix: np.ndarray, k_values: range, seed: int) -> pd.DataFrame:
    """Fit K-Means for each k and collect the quality measures."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(matrix)
        rows.append({"k": k, "inertia": float(model.inertia_), **_cluster_quality(matrix, model.labels_, seed)})
        logger.info("KMeans k=%s | silhouette %.3f | Davies-Bouldin %.3f", k, rows[-1]["silhouette"], rows[-1]["davies_bouldin"])
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame, min_clusters: int, max_clusters: int) -> int:
    """Highest silhouette among ``min_clusters <= k <= max_clusters``.

    Two clusters are too coarse for marketing and many clusters are hard to act on,
    so the choice is limited to a practical range (falls back to all k if none qualify).
    """
    eligible = scores[(scores["k"] >= min_clusters) & (scores["k"] <= max_clusters)]
    pool = eligible if not eligible.empty else scores
    return int(pool.loc[pool["silhouette"].idxmax(), "k"])
    

def run_gmm(matrix: np.ndarray, n_components: int, seed: int, kmeans_labels: np.ndarray) -> tuple[GaussianMixture, dict[str, float]]:
    cfg = get_config_value("segmentation.gmm", {}) or {}
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=str(cfg.get("covariance_type", "full")),
        n_init=int(cfg.get("n_init", 3)),
        random_state=seed,
    ).fit(matrix)
    labels = model.predict(matrix)
    metrics = {
        **_cluster_quality(matrix, labels, seed),
        "bic": float(model.bic(matrix)),
        "agreement_with_kmeans_ari": float(adjusted_rand_score(kmeans_labels, labels)),
    }
    return model, metrics


def run_dbscan(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Density clustering used mainly to find unusual (noise) customers.

    ``eps`` comes from config or, when absent, from the 95th percentile of each
    customer's distance to its ``min_samples``-th nearest neighbour.
    """
    cfg = get_config_value("segmentation.dbscan", {}) or {}
    min_samples = int(cfg.get("min_samples", 10))
    eps = cfg.get("eps")
    if eps is None:
        distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(matrix).kneighbors(matrix)
        eps = float(np.percentile(distances[:, -1], 95))
    labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, {"eps": float(eps), "min_samples": min_samples, "n_clusters": int(n_clusters), "noise_share": float((labels == -1).mean())}


def find_outliers(matrix: np.ndarray, neighbours: int = 10, percentile: float = 95.0) -> np.ndarray:
    """Flag the customers that are furthest from their nearest neighbours (top 5% by default)."""
    distances, _ = NearestNeighbors(n_neighbors=neighbours).fit(matrix).kneighbors(matrix)
    return distances[:, -1] > np.percentile(distances[:, -1], percentile)


def run_hierarchical(matrix: np.ndarray, n_clusters: int, kmeans_labels: np.ndarray, seed: int) -> tuple[dict[str, float], np.ndarray]:
    """Ward linkage on a random sample (it needs O(n^2) memory, so it is not run on everyone)."""
    size = min(int(get_config_value("segmentation.hierarchical_sample_size", 500)), len(matrix))
    sample = np.random.default_rng(seed).choice(len(matrix), size=size, replace=False)
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(matrix[sample])
    metrics = {
        "sample_size": int(size),
        "silhouette": float(silhouette_score(matrix[sample], labels)),
        "agreement_with_kmeans_ari": float(adjusted_rand_score(kmeans_labels[sample], labels)),
    }
    return metrics, sample

# --------------------------------------------------------------------------- #
# Profiling and naming
# --------------------------------------------------------------------------- #
def _name_segment(
    recency_score: float,
    frequency_score: float,
    monetary_score: float,
    recency_days: float,
    discount_share: float,
    seasonal_share: float,
) -> str:
    """Business label for one cluster from its average behaviour.

    The three ``*_score`` values are average percentile ranks (0-1; a high recency
    score means the customer bought recently). Distinctive buying styles (heavy
    discount use, buying only in the peak season) are checked first.
    """
    if discount_share >= 0.4:
        return "Discount Bulk Buyers"
    if seasonal_share >= 0.8:
        return "Seasonal Shoppers"
    if recency_score >= 0.6 and frequency_score >= 0.6 and monetary_score >= 0.6:
        return "Champions"
    if recency_score >= 0.5 and frequency_score >= 0.5:
        return "Loyal Customers"
    if recency_score >= 0.6:
        return "New & Promising"
    if recency_days >= 330:
        return "Lost Customers"
    if frequency_score >= 0.5 or monetary_score >= 0.5:
        return "At Risk"
    return "Lapsing Occasional Buyers"


def profile_segments(table: pd.DataFrame, labels: np.ndarray, features: list[str]) -> tuple[pd.DataFrame, dict[int, str]]:
    """Average feature values, size, churn rate and revenue per cluster, plus segment names."""
    data = table.assign(cluster=labels)
    scores = pd.DataFrame(
        {
            "r": table["recency_days"].rank(pct=True, ascending=False),
            "f": table["frequency"].rank(pct=True),
            "m": table["monetary_total"].rank(pct=True),
            "cluster": labels,
        }
    )
    score_means = scores.groupby("cluster")[["r", "f", "m"]].mean()
    raw_means = data.groupby("cluster")[features].mean()

    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for cluster, row in score_means.sort_values(["m", "f"], ascending=False).iterrows():
        raw = raw_means.loc[cluster]
        base = _name_segment(
            row["r"],
            row["f"],
            row["m"],
            recency_days=float(raw["recency_days"]),
            discount_share=float(raw.get("discount_line_share", 0.0)),
            seasonal_share=float(raw.get("seasonal_order_share", 0.0)),
        )
        used[base] = used.get(base, 0) + 1
        names[int(cluster)] = base if used[base] == 1 else f"{base} ({used[base]})"

    profile = raw_means.copy()
    profile.insert(0, "customers", data.groupby("cluster").size())
    profile.insert(1, "share_pct", 100 * profile["customers"] / len(data))
    if CHURN_TARGET in data:
        profile["churn_rate"] = data.groupby("cluster")[CHURN_TARGET].mean()
    if CLV_TARGET in data:
        profile["avg_future_revenue"] = data.groupby("cluster")[CLV_TARGET].mean()
    profile.insert(0, "segment", [names[int(c)] for c in profile.index])
    return profile.round(3), names

# --------------------------------------------------------------------------- #
# Plots (optional: skipped quietly if matplotlib is missing)
# --------------------------------------------------------------------------- #
def save_plots(
    figures_dir,
    scores: pd.DataFrame,
    chosen_k: int,
    matrix: np.ndarray,
    labels: np.ndarray,
    names: dict[int, str],
    sample_matrix: np.ndarray,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
    except ImportError:
        logger.warning("matplotlib/scipy not available - skipping plots")
        return []

    saved = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set(title="Elbow curve", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="tab:green")
    axes[1].axvline(chosen_k, linestyle="--", color="grey")
    axes[1].set(title="Silhouette score", xlabel="Number of clusters (k)", ylabel="Silhouette")
    fig.tight_layout()
    path = figures_dir / "kmeans_k_selection.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(str(path))

    coords = PCA(n_components=2, random_state=0).fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster in sorted(names):
        mask = labels == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=names[cluster])
    ax.set(title="Customer segments (PCA projection)", xlabel="Component 1", ylabel="Component 2")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    path = figures_dir / "segments_pca.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 4))
    dendrogram(linkage(sample_matrix, method="ward"), no_labels=True, truncate_mode="lastp", p=30, ax=ax)
    ax.set(title="Hierarchical clustering (sample, truncated)", ylabel="Distance")
    fig.tight_layout()
    path = figures_dir / "hierarchical_dendrogram.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(str(path))
    return saved


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save: bool = True) -> pd.DataFrame:
    seed = get_seed()
    table = load_feature_table()
    matrix, features, preprocessor = prepare_matrix(table)
    logger.info("Segmenting %s customers on %s features", f"{len(table):,}", len(features))

    k_range = get_config_value("segmentation.kmeans.k_range", DEFAULT_K_RANGE)
    k_values = range(int(k_range[0]), int(k_range[1]) + 1)
    min_clusters = int(get_config_value("segmentation.min_clusters", 4))
    max_clusters = int(get_config_value("segmentation.max_clusters", 6))
    scores = evaluate_kmeans(matrix, k_values, seed)
    best_k = choose_k(scores, min_clusters, max_clusters)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(matrix)
    labels = kmeans.labels_
    logger.info("Chosen k = %s", best_k)

    second = str(get_config_value("segmentation.second_algorithm", "gmm")).lower()
    comparison: dict[str, Any] = {}
    gmm_labels = None
    if second == "gmm":
        gmm, comparison["gmm"] = run_gmm(matrix, best_k, seed, labels)
        gmm_labels = gmm.predict(matrix)
    dbscan_labels, comparison["dbscan"] = run_dbscan(matrix)
    comparison["hierarchical"], sample = run_hierarchical(matrix, best_k, labels, seed)

    profile, names = profile_segments(table, labels, features)

    output = pd.DataFrame(
        {
            ID_COLUMN: table[ID_COLUMN],
            "segment_id": labels,
            "segment_name": [names[int(c)] for c in labels],
            "is_outlier": find_outliers(matrix),
        }
    )
    if gmm_labels is not None:
        output["gmm_segment_id"] = gmm_labels

    if save:
        artifacts_dir = resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"), create=True)
        segments_path = resolve_path(
            get_config_value("paths.customer_segments", "data/processed/customer_segments.parquet"), create=True
        )
        figures_dir = resolve_path(get_config_value("paths.figures_dir", "docs/figures"), create=True)

        output.to_parquet(segments_path, index=False)
        profile.to_csv(artifacts_dir / "segment_profiles.csv")
        scores.to_csv(artifacts_dir / "segmentation_k_scores.csv", index=False)
        joblib.dump(
            {"preprocessor": preprocessor, "kmeans": kmeans, "features": features, "segment_names": names},
            artifacts_dir / "segmentation_kmeans.joblib",
        )
        plots = save_plots(figures_dir, scores, best_k, matrix, labels, names, matrix[sample])
        metadata = {
            "chosen_k": best_k,
            "min_clusters": min_clusters,
            "features": features,
            "kmeans": scores.loc[scores["k"] == best_k].iloc[0].to_dict(),
            "comparison": comparison,
            "segment_names": {str(k): v for k, v in names.items()},
            "plots": plots,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        (artifacts_dir / "segmentation_metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
        logger.info("Saved segments to %s", segments_path)

    _report(scores, best_k, comparison, profile)
    return output


def _report(scores: pd.DataFrame, best_k: int, comparison: dict[str, Any], profile: pd.DataFrame) -> None:
    print("\nK-Means quality by number of clusters")
    print(scores.round(3).to_string(index=False))
    print(f"\nChosen k: {best_k}")
    print("\nOther algorithms")
    for name, metrics in comparison.items():
        print(f"  {name}: " + ", ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()))
    print("\nSegment profiles (feature averages)")
    print(profile.to_string())


if __name__ == "__main__":
    argparse.ArgumentParser(description="Customer segmentation").parse_args()
    run()