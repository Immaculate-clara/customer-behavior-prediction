"""Cleaning of raw transactions into the interim dataset.

Pipeline: validate null rates -> remove sheet-overlap duplicates -> drop rows
with missing keys -> flag cancellations -> drop non-product codes -> drop
invalid quantity/price rows -> (optionally) drop guest rows with no customer
-> flag outliers -> add line_total -> save Parquet.
"""
from __future__ import annotations

import pandas as pd

from src.data.loader import load_transactions
from src.utils.config import get_config_value, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price"]


def _log_step(step: str, before: int, after: int) -> None:
    logger.info("%-42s removed %9s rows | remaining %9s", step, f"{before - after:,}", f"{after:,}")


def _validate_null_rates(df: pd.DataFrame) -> None:
    """Fail fast if a column has more missing values than the config allows."""
    limits: dict[str, float] = get_config_value("data.max_null_rate", {})
    for column, limit in limits.items():
        if column not in df.columns:
            continue
        rate = float(df[column].isna().mean())
        logger.info("Null rate for %-12s %.2f%% (limit %.0f%%)", column, rate * 100, limit * 100)
        if rate > limit:
            raise ValueError(
                f"Column '{column}' has {rate:.1%} missing values, above the allowed {limit:.0%}"
            )


def _remove_sheet_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows duplicated because consecutive sheets cover overlapping days.

    Any invoice already seen in an earlier sheet is removed from later sheets.
    Repeated lines inside a single invoice are genuine and are kept.
    """
    seen: set[str] = set()
    keep = pd.Series(True, index=df.index)
    for sheet in get_config_value("data.sheet_names", []):
        in_sheet = df["source_sheet"] == sheet
        keep &= ~(in_sheet & df["invoice"].isin(seen))
        seen |= set(df.loc[in_sheet, "invoice"].dropna())
    return df[keep]


def _flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Add outlier flags without removing any rows."""
    multiplier = float(get_config_value("data.outlier.iqr_multiplier", 3.0))
    bulk_threshold = int(get_config_value("data.outlier.quantity_bulk_threshold", 1000))

    sales_quantity = df.loc[~df["is_cancellation"], "quantity"]
    q1, q3 = sales_quantity.quantile([0.25, 0.75])
    upper_fence = q3 + multiplier * (q3 - q1)

    df["is_bulk_order"] = df["quantity"].abs() >= bulk_threshold
    df["is_quantity_outlier"] = df["quantity"].abs() > upper_fence
    logger.info(
        "Flagged %s bulk-order rows and %s quantity outliers (fence=%.1f)",
        f"{int(df['is_bulk_order'].sum()):,}",
        f"{int(df['is_quantity_outlier'].sum()):,}",
        upper_fence,
    )
    return df


def clean_transactions(
    df: pd.DataFrame | None = None,
    drop_missing_customers: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """Clean raw transactions and (optionally) save them to ``data/interim``.

    Args:
        df: Raw transactions. Loaded with ``load_transactions`` when omitted.
        drop_missing_customers: Drop rows without a customer_id (needed for
            customer-level modelling).
        save: Write the result to ``paths.interim_transactions``.
    """
    df = load_transactions() if df is None else df.copy()
    logger.info("Starting cleaning with %s rows", f"{len(df):,}")
    _validate_null_rates(df)

    before = len(df)
    df = _remove_sheet_overlap(df)
    _log_step("Overlapping sheet duplicates", before, len(df))

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["quantity"] = df["quantity"].astype("int64")
    _log_step("Missing invoice/stock/qty/date/price", before, len(df))

    df["is_cancellation"] = df["invoice"].str.startswith("C").fillna(False).astype(bool)

    before = len(df)
    non_product = {code.upper() for code in get_config_value("data.non_product_codes", [])}
    df = df[~df["stock_code"].isin(non_product)]
    _log_step("Non-product stock codes", before, len(df))

    before = len(df)
    valid_sale = ~df["is_cancellation"] & (df["quantity"] > 0) & (df["unit_price"] > 0)
    valid_return = df["is_cancellation"] & (df["quantity"] < 0) & (df["unit_price"] > 0)
    df = df[valid_sale | valid_return].copy()
    _log_step("Invalid quantity or price", before, len(df))

    if drop_missing_customers:
        before = len(df)
        df = df.dropna(subset=["customer_id"]).copy()
        df["customer_id"] = df["customer_id"].astype("int64")
        _log_step("Missing customer_id", before, len(df))

    df = _flag_outliers(df)
    df["line_total"] = df["quantity"] * df["unit_price"]

    df = df.sort_values(["invoice_date", "invoice"]).reset_index(drop=True)

    logger.info(
        "Cleaning finished: %s rows, %s customers, %s cancellation rows",
        f"{len(df):,}",
        f"{df['customer_id'].nunique():,}",
        f"{int(df['is_cancellation'].sum()):,}",
    )

    if save:
        output_path = resolve_path(
            get_config_value("paths.interim_transactions", "data/interim/transactions_clean.parquet"),
            create=True,
        )
        df.to_parquet(output_path, index=False)
        logger.info("Saved cleaned transactions to %s", output_path)

    return df


if __name__ == "__main__":
    cleaned = clean_transactions()
    print(cleaned.head())
    print(cleaned.dtypes)