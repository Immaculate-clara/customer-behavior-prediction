"""Validation checks for the cleaned transactions (run before feature engineering).

Checks schema, dtypes, null rates, date range, value sanity and cross-sheet
duplicates. Fails loudly with ``DataValidationError`` when any check fails.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pdt

from src.utils.config import get_config_value, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "invoice", "stock_code", "description", "quantity", "invoice_date",
    "unit_price", "customer_id", "country", "source_sheet",
    "is_cancellation", "is_bulk_order", "is_quantity_outlier", "line_total",
]
KEY_COLUMNS = ["invoice", "stock_code", "quantity", "invoice_date", "unit_price", "customer_id"]


class DataValidationError(Exception):
    """Raised when the cleaned data fails one or more validation checks."""


@dataclass
class ValidationReport:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed.append(name)
            logger.info("PASS | %s", name)
        else:
            message = f"{name} - {detail}" if detail else name
            self.failed.append(message)
            logger.error("FAIL | %s", message)


def load_clean_transactions() -> pd.DataFrame:
    """Read the cleaned transactions produced by the cleaning stage."""
    path = resolve_path(
        get_config_value("paths.interim_transactions", "data/interim/transactions_clean.parquet")
    )
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}. Run 'python -m src.data.cleaner' first.")
    return pd.read_parquet(path)


def validate_clean_transactions(
    df: pd.DataFrame | None = None, raise_on_error: bool = True
) -> ValidationReport:
    """Run all checks on the cleaned transactions and return a report."""
    df = load_clean_transactions() if df is None else df
    report = ValidationReport()

    # 1. Schema
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    report.check("Schema: all expected columns present", not missing, f"missing {missing}")
    report.check("Dataset is not empty", len(df) > 0, "0 rows")
    if missing or df.empty:
        return _finish(report, raise_on_error)

    # 2. Data types
    report.check("Dtype: quantity is integer", pdt.is_integer_dtype(df["quantity"]))
    report.check("Dtype: customer_id is integer", pdt.is_integer_dtype(df["customer_id"]))
    report.check("Dtype: unit_price is numeric", pdt.is_float_dtype(df["unit_price"]))
    report.check("Dtype: invoice_date is datetime", pdt.is_datetime64_any_dtype(df["invoice_date"]))

    # 3. Null rates
    key_nulls = {col: int(df[col].isna().sum()) for col in KEY_COLUMNS if df[col].isna().any()}
    report.check("Nulls: no missing values in key columns", not key_nulls, f"{key_nulls}")
    description_limit = float(get_config_value("data.max_null_rate.description", 0.02))
    description_rate = float(df["description"].isna().mean())
    report.check(
        "Nulls: description within allowed rate",
        description_rate <= description_limit,
        f"{description_rate:.2%} > {description_limit:.2%}",
    )

    # 4. Date range
    expected = get_config_value("data.expected_date_range")
    if expected:
        low = pd.Timestamp(expected[0])
        high = pd.Timestamp(expected[1]) + pd.Timedelta(days=1)
        outside = int(((df["invoice_date"] < low) | (df["invoice_date"] >= high)).sum())
        report.check("Dates: all invoices inside expected range", outside == 0, f"{outside:,} rows outside {expected}")

    # 5. Value sanity
    report.check("Values: unit_price > 0 everywhere", bool((df["unit_price"] > 0).all()))
    report.check("Values: customer_id > 0 everywhere", bool((df["customer_id"] > 0).all()))
    sales_ok = bool((df.loc[~df["is_cancellation"], "quantity"] > 0).all())
    returns_ok = bool((df.loc[df["is_cancellation"], "quantity"] < 0).all())
    report.check("Values: sales have positive quantity", sales_ok)
    report.check("Values: cancellations have negative quantity", returns_ok)

    # 6. Cross-sheet duplicates
    sheets_per_invoice = df.groupby("invoice")["source_sheet"].nunique()
    report.check(
        "Duplicates: no invoice appears in more than one sheet",
        bool((sheets_per_invoice <= 1).all()),
        f"{int((sheets_per_invoice > 1).sum()):,} invoices repeated",
    )

    return _finish(report, raise_on_error)


def _finish(report: ValidationReport, raise_on_error: bool) -> ValidationReport:
    logger.info("Validation summary: %s passed, %s failed", len(report.passed), len(report.failed))
    if report.failed and raise_on_error:
        raise DataValidationError("Validation failed:\n- " + "\n- ".join(report.failed))
    return report


if __name__ == "__main__":
    try:
        validate_clean_transactions()
    except (DataValidationError, FileNotFoundError) as error:
        print(error)
        sys.exit(1)
    print("All validation checks passed.")