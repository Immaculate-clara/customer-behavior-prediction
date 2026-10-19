"""Raw data loading for the Online Retail II dataset.

Reads both Excel sheets, standardises column names and dtypes, and caches
the combined result as Parquet so later runs skip the slow Excel parse.
Cleaning (cancellations, duplicates, outliers) happens in a later stage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import get_config_value, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_CACHE_NAME = "transactions_raw.parquet"
STRING_COLUMNS = ["invoice", "stock_code", "description", "country"]


def _raw_cache_path() -> Path:
    """Location of the cached raw-combined Parquet file."""
    interim_dir = resolve_path(get_config_value("paths.interim_dir", "data/interim"), create=True)
    return interim_dir / RAW_CACHE_NAME


def _read_excel_sheets(path: Path, sheet_names: list[str]) -> pd.DataFrame:
    """Read every configured sheet and stack them into one DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    frames = []
    for sheet in sheet_names:
        logger.info("Reading sheet '%s' (this can take a few minutes)...", sheet)
        frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        frame["source_sheet"] = sheet
        logger.info("Sheet '%s': %s rows", sheet, f"{len(frame):,}")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using the config map and enforce consistent dtypes."""
    column_map: dict[str, str] = get_config_value("data.column_map", {})
    if not column_map:
        raise ValueError("'data.column_map' is missing from config.yaml")

    missing = [col for col in column_map if col not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in the Excel file: {missing}")

    df = df.rename(columns=column_map)
    df = df[[*column_map.values(), "source_sheet"]].copy()

    for col in STRING_COLUMNS:
        df[col] = df[col].astype("string").str.strip()
    df["stock_code"] = df["stock_code"].str.upper()

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").astype("float64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")

    return df


def _log_summary(df: pd.DataFrame) -> None:
    """Log basic facts and warn when the data looks different from expectations."""
    logger.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])
    logger.info(
        "Date range: %s to %s", df["invoice_date"].min(), df["invoice_date"].max()
    )
    logger.info("Missing customer_id: %.1f%%", df["customer_id"].isna().mean() * 100)

    expected = get_config_value("data.expected_date_range")
    if expected:
        low = pd.Timestamp(expected[0])
        high = pd.Timestamp(expected[1]) + pd.Timedelta(days=1)
        outside = int(((df["invoice_date"] < low) | (df["invoice_date"] >= high)).sum())
        if outside:
            logger.warning("%s rows fall outside the expected date range %s", f"{outside:,}", expected)

    if df["invoice_date"].isna().any():
        logger.warning("%s rows have an unparseable invoice_date", f"{int(df['invoice_date'].isna().sum()):,}")


def load_transactions(use_cache: bool = True, force_reload: bool = False) -> pd.DataFrame:
    """Return the combined, column-standardised raw transactions.

    Args:
        use_cache: Read/write the Parquet cache in ``data/interim``.
        force_reload: Ignore the cache and re-read the Excel file.
    """
    excel_path = resolve_path(get_config_value("paths.raw_data", "data/raw/online_retail_II.xlsx"))
    cache_path = _raw_cache_path()

    cache_is_fresh = (
        use_cache
        and not force_reload
        and cache_path.exists()
        and excel_path.exists()
        and cache_path.stat().st_mtime >= excel_path.stat().st_mtime
    )
    if cache_is_fresh:
        logger.info("Loading cached raw data from %s", cache_path)
        return pd.read_parquet(cache_path)

    sheet_names = get_config_value("data.sheet_names", [])
    if not sheet_names:
        raise ValueError("'data.sheet_names' is missing from config.yaml")

    df = _standardize(_read_excel_sheets(excel_path, sheet_names))
    _log_summary(df)

    if use_cache:
        df.to_parquet(cache_path, index=False)
        logger.info("Cached raw data to %s", cache_path)

    return df


if __name__ == "__main__":
    data = load_transactions()
    print(data.head())
    print(data.dtypes)