"""Customer-level feature engineering with point-in-time (leakage-safe) discipline.

Every feature is computed only from transactions strictly BEFORE the cutoff
date. Targets (churn and future revenue) are computed only from the
``horizon_days`` window that starts AT the cutoff date.

    history: invoice_date <  cutoff
    future : cutoff <= invoice_date < cutoff + horizon_days
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_config_value, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

SECONDS_PER_DAY = 86_400.0
TREND_BUCKET_DAYS = 30


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def _cutoff_from_config() -> pd.Timestamp:
    value = get_config_value("features.cutoff_date")
    if not value:
        raise ValueError("'features.cutoff_date' is missing from config.yaml")
    return pd.Timestamp(value)


def _horizon_from_config() -> int:
    return int(get_config_value("features.churn_horizon_days", 90))


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def split_history(transactions: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Return only rows strictly before the cutoff (the only data features may see)."""
    history = transactions[transactions["invoice_date"] < cutoff]
    if history.empty:
        raise ValueError(f"No transactions before the cutoff date {cutoff.date()}")
    return history


def _invoice_table(sales: pd.DataFrame) -> pd.DataFrame:
    """Collapse sales lines to one row per (customer, invoice), sorted by time."""
    invoices = sales.groupby(["customer_id", "invoice"], as_index=False).agg(
        invoice_date=("invoice_date", "min"),
        revenue=("line_total", "sum"),
        units=("quantity", "sum"),
        n_lines=("stock_code", "nunique"),
    )
    return invoices.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)


def _frequency_trend(invoices: pd.DataFrame, cutoff: pd.Timestamp, window: int) -> pd.Series:
    """Slope of orders-per-30-day-bucket over the last ``window`` buckets.

    Positive = customer is ordering more often recently; negative = slowing down.
    """
    bucket = (cutoff - invoices["invoice_date"]).dt.days // TREND_BUCKET_DAYS
    in_window = bucket < window
    recent = invoices.loc[in_window, ["customer_id"]].assign(
        period=(window - 1 - bucket[in_window]).astype(int)  # 0 = oldest, window-1 = newest
    )
    counts = (
        recent.groupby(["customer_id", "period"]).size().unstack(fill_value=0)
        .reindex(columns=range(window), fill_value=0)
    )
    x_centered = np.arange(window) - (window - 1) / 2
    slope = (counts.to_numpy() * x_centered).sum(axis=1) / (x_centered**2).sum()
    return pd.Series(slope, index=counts.index, name="purchase_frequency_trend")


def _interpurchase_stats(invoices: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of days between consecutive orders."""
    gaps = invoices.groupby("customer_id")["invoice_date"].diff().dt.total_seconds() / SECONDS_PER_DAY
    return (
        gaps.groupby(invoices["customer_id"])
        .agg(interpurchase_mean_days="mean", interpurchase_std_days="std")
    )


def _discount_sensitivity(sales: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Proxy for markdown sensitivity (the dataset has no discount field).

    Each line's price is compared with that product's median price in the
    history window. ``discount_line_share`` is the share of a customer's lines
    bought below ``threshold`` x the median price.
    """
    median_price = sales.groupby("stock_code")["unit_price"].median()
    price_index = sales["unit_price"] / sales["stock_code"].map(median_price)
    lines = pd.DataFrame(
        {
            "customer_id": sales["customer_id"],
            "price_index": price_index,
            "is_discounted": price_index < threshold,
        }
    )
    return lines.groupby("customer_id").agg(
        discount_line_share=("is_discounted", "mean"),
        avg_price_index=("price_index", "mean"),
    )


def _engagement_score(features: pd.DataFrame) -> pd.Series:
    """Composite 0-100 score from percentile ranks of recency, frequency and spend."""
    recency_score = features["recency_days"].rank(pct=True, ascending=False)
    frequency_score = features["frequency"].rank(pct=True)
    monetary_score = features["monetary_total"].rank(pct=True)
    return 100 * (recency_score + frequency_score + monetary_score) / 3


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_customer_features(
    transactions: pd.DataFrame,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build one row of features per customer using data before ``cutoff_date``."""
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else _cutoff_from_config()
    seasonal_months = list(get_config_value("features.seasonal_months", [10, 11, 12]))
    trend_window = int(get_config_value("features.trend_window_months", 12))
    discount_threshold = float(get_config_value("features.discount_threshold", 0.9))
    min_orders = int(get_config_value("features.min_orders_for_model", 1))

    history = split_history(transactions, cutoff)
    sales = history[~history["is_cancellation"]]
    returns = history[history["is_cancellation"]]
    invoices = _invoice_table(sales)

    # Recency / Frequency / Monetary, basket and tenure
    features = invoices.groupby("customer_id").agg(
        frequency=("invoice", "nunique"),
        monetary_total=("revenue", "sum"),
        monetary_avg=("revenue", "mean"),
        avg_basket_size=("units", "mean"),
        avg_basket_lines=("n_lines", "mean"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    )
    features["recency_days"] = (cutoff - features["last_purchase"]).dt.total_seconds() / SECONDS_PER_DAY
    features["tenure_days"] = (cutoff - features["first_purchase"]).dt.total_seconds() / SECONDS_PER_DAY
    features = features.drop(columns=["first_purchase", "last_purchase"])

    # Timing behaviour
    features = features.join(_interpurchase_stats(invoices))
    trend = _frequency_trend(invoices, cutoff, trend_window)
    features["purchase_frequency_trend"] = trend.reindex(features.index).fillna(0.0)

    # Seasonality: share of orders placed in the configured peak months
    is_seasonal = invoices["invoice_date"].dt.month.isin(seasonal_months)
    features["seasonal_order_share"] = is_seasonal.groupby(invoices["customer_id"]).mean()

    # Returns
    return_stats = returns.groupby("customer_id").agg(
        n_returns=("invoice", "nunique"), returned_value=("line_total", "sum")
    )
    features = features.join(return_stats)
    features["return_rate"] = (features["returned_value"].abs() / features["monetary_total"]).clip(0, 1)
    features["historical_clv"] = features["monetary_total"] + features["returned_value"].fillna(0.0)

    # Discount sensitivity
    features = features.join(_discount_sensitivity(sales, discount_threshold))

    # Fill structural gaps: no returns, or too few orders to measure gaps
    features["n_returns"] = features["n_returns"].fillna(0).astype(int)
    fill_zero = ["returned_value", "return_rate", "interpurchase_mean_days", "interpurchase_std_days"]
    features[fill_zero] = features[fill_zero].fillna(0.0)
    features = features.drop(columns=["returned_value"])

    features["engagement_score"] = _engagement_score(features)

    features = features[features["frequency"] >= min_orders]
    logger.info(
        "Built %s features for %s customers (history before %s)",
        features.shape[1], f"{len(features):,}", cutoff.date(),
    )
    return features


def build_labels(
    transactions: pd.DataFrame,
    customer_ids: pd.Index,
    cutoff_date: str | pd.Timestamp | None = None,
    horizon_days: int | None = None,
) -> pd.DataFrame:
    """Build churn and future-revenue targets from the window after the cutoff.

    ``churn`` = 1 when the customer places no order in the horizon window.
    ``future_revenue`` = sales revenue in that window (the CLV regression target).
    """
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date is not None else _cutoff_from_config()
    horizon = int(horizon_days) if horizon_days is not None else _horizon_from_config()
    window_end = cutoff + pd.Timedelta(days=horizon)

    if transactions["invoice_date"].max() < window_end - pd.Timedelta(days=1):
        raise ValueError(
            f"Data ends {transactions['invoice_date'].max().date()}, before the full "
            f"{horizon}-day horizon ends ({window_end.date()}). Choose an earlier cutoff."
        )

    future = transactions[
        (transactions["invoice_date"] >= cutoff)
        & (transactions["invoice_date"] < window_end)
        & (~transactions["is_cancellation"])
    ]
    revenue = future.groupby("customer_id")["line_total"].sum()
    purchasers = future["customer_id"].unique()

    labels = pd.DataFrame(index=customer_ids)
    labels["churn"] = (~customer_ids.isin(purchasers)).astype(int)
    labels["future_revenue"] = revenue.reindex(customer_ids).fillna(0.0)
    return labels


def build_feature_table(
    transactions: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Create the customer-level modelling table (features + targets)."""
    if transactions is None:
        path = resolve_path(
            get_config_value("paths.interim_transactions", "data/interim/transactions_clean.parquet")
        )
        if not path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {path}. Run the cleaner first.")
        transactions = pd.read_parquet(path)

        from src.features.affinity import build_affinity_features, build_next_category_labels

    cutoff = _cutoff_from_config()
    features = build_customer_features(transactions, cutoff)
    features = features.join(build_affinity_features(transactions, cutoff))
    labels = build_labels(transactions, features.index, cutoff)
    labels["next_category"] = build_next_category_labels(transactions, features.index, cutoff)
    table = features.join(labels).reset_index()

    logger.info(
        "Feature table: %s customers | churn rate %.1f%% | mean future revenue %.2f",
        f"{len(table):,}", table["churn"].mean() * 100, table["future_revenue"].mean(),
    )

    if save:
        output_path = resolve_path(
            get_config_value("paths.customer_features", "data/processed/customer_features.parquet"),
            create=True,
        )
        table.to_parquet(output_path, index=False)
        logger.info("Saved customer features to %s", output_path)

    return table


if __name__ == "__main__":
    result = build_feature_table()
    print(result.head())
    print(result.describe().T)