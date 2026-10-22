"""Tests for raw-data loading, cleaning and validation (no real dataset needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import cleaner, loader, validator
from src.data.validator import DataValidationError, ValidationReport, validate_clean_transactions

COLUMN_MAP = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def fake_config(values: dict):
    """Replacement for get_config_value that reads from a flat dict of dotted keys."""

    def getter(key, default=None, path=None):
        return values.get(key, default)

    return getter


def fake_resolve(base: Path):
    """Replacement for resolve_path that keeps every file inside a temporary folder."""

    def resolver(relative_path, create=False):
        path = base / Path(relative_path).name
        if create:
            (path if not path.suffix else path.parent).mkdir(parents=True, exist_ok=True)
        return path

    return resolver


# ======================================================================= cleaner
CLEAN_CONFIG = {
    "data.max_null_rate": {"description": 0.5},
    "data.sheet_names": ["A", "B"],
    "data.non_product_codes": ["post"],
    "data.outlier.iqr_multiplier": 3.0,
    "data.outlier.quantity_bulk_threshold": 1000,
}


@pytest.fixture()
def raw() -> pd.DataFrame:
    rows = [
        # invoice, stock, description, qty, price, customer, sheet
        ("1001", "M1", "MUG", 2, 5.0, 1.0, "A"),       # valid sale
        ("1001", "M1", "MUG", 1, 5.0, 1.0, "A"),       # repeated line in the same invoice: kept
        ("1001", "M1", "MUG", 2, 5.0, 1.0, "B"),       # same invoice in a later sheet: overlap, removed
        ("1002", "M2", "PEN", 3, 2.0, np.nan, "B"),    # guest customer
        ("C1003", "M1", "MUG", -1, 5.0, 1.0, "B"),     # valid return
        ("1004", "POST", "POSTAGE", 1, 9.0, 1.0, "B"),  # non-product code
        ("1005", "M3", "CUP", -5, 4.0, 1.0, "B"),      # sale with negative quantity
        ("1006", "M3", "CUP", 5, 0.0, 1.0, "B"),       # zero price
        ("1007", None, "BOWL", 5, 3.0, 1.0, "B"),      # missing stock code
        ("1008", "M4", "BULK", 5000, 1.0, 2.0, "B"),   # bulk order
    ]
    frame = pd.DataFrame(
        rows,
        columns=["invoice", "stock_code", "description", "quantity", "unit_price", "customer_id", "source_sheet"],
    )
    frame["invoice_date"] = pd.to_datetime(
        ["2011-01-05", "2011-01-05", "2011-01-06", "2011-02-01", "2011-03-01",
         "2011-03-02", "2011-03-03", "2011-03-04", "2011-03-05", "2011-04-01"]
    )
    frame["country"] = "United Kingdom"
    return frame


@pytest.fixture()
def clean_config(monkeypatch):
    monkeypatch.setattr(cleaner, "get_config_value", fake_config(CLEAN_CONFIG))


def test_cleaning_removes_invalid_rows_and_keeps_valid_ones(raw, clean_config):
    cleaned = cleaner.clean_transactions(raw, save=False)

    assert list(cleaned["invoice"]) == ["1001", "1001", "C1003", "1008"]
    assert cleaned["customer_id"].dtype == "int64"
    assert cleaned["invoice_date"].is_monotonic_increasing


def test_cleaning_flags_cancellations_and_adds_line_totals(raw, clean_config):
    cleaned = cleaner.clean_transactions(raw, save=False)

    returns = cleaned[cleaned["is_cancellation"]]
    assert list(returns["invoice"]) == ["C1003"]
    assert (cleaned["line_total"] == cleaned["quantity"] * cleaned["unit_price"]).all()
    assert returns["line_total"].iloc[0] == -5.0


def test_cleaning_flags_bulk_orders_without_removing_them(raw, clean_config):
    cleaned = cleaner.clean_transactions(raw, save=False)
    assert cleaned.loc[cleaned["invoice"] == "1008", "is_bulk_order"].all()
    assert not cleaned.loc[cleaned["invoice"] == "1001", "is_bulk_order"].any()


def test_outlier_flags_mark_unusually_large_quantities(clean_config):
    quantities = [1, 2, 2, 3, 3, 4, 5, 5, 6, 5000]
    frame = pd.DataFrame({"quantity": quantities, "is_cancellation": False})

    flagged = cleaner._flag_outliers(frame)

    assert flagged["is_bulk_order"].tolist() == [False] * 9 + [True]
    assert flagged["is_quantity_outlier"].tolist() == [False] * 9 + [True]


def test_cleaning_can_keep_guest_customers(raw, clean_config):
    cleaned = cleaner.clean_transactions(raw, drop_missing_customers=False, save=False)
    assert "1002" in set(cleaned["invoice"])
    assert cleaned["customer_id"].isna().sum() == 1


def test_cleaning_does_not_modify_the_input_frame(raw, clean_config):
    before = raw.copy()
    cleaner.clean_transactions(raw, save=False)
    pd.testing.assert_frame_equal(raw, before)


def test_cleaning_saves_the_interim_file(raw, clean_config, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(cleaner, "resolve_path", fake_resolve(tmp_path))
    cleaner.clean_transactions(raw, save=True)
    assert (tmp_path / "transactions_clean.parquet").exists()


def test_cleaning_loads_raw_data_when_no_frame_is_given(raw, clean_config, monkeypatch):
    monkeypatch.setattr(cleaner, "load_transactions", lambda: raw)
    assert len(cleaner.clean_transactions(save=False)) == 4


def test_null_rate_check_rejects_too_many_missing_values(raw, monkeypatch):
    monkeypatch.setattr(cleaner, "get_config_value", fake_config({"data.max_null_rate": {"description": 0.1}}))
    raw.loc[:4, "description"] = None
    with pytest.raises(ValueError, match="missing values"):
        cleaner.clean_transactions(raw, save=False)


def test_null_rate_check_ignores_columns_that_are_absent(raw, monkeypatch):
    monkeypatch.setattr(cleaner, "get_config_value", fake_config({"data.max_null_rate": {"not_a_column": 0.0}}))
    cleaner._validate_null_rates(raw)


# ======================================================================= loader
@pytest.fixture()
def excel_like() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Invoice": [" 1001", "C1002"],
            "StockCode": [" m1 ", "85123a"],
            "Description": ["MUG ", " STAR"],
            "Quantity": ["2", "abc"],
            "InvoiceDate": ["2011-01-05 10:00", "not a date"],
            "Price": ["5.5", "2"],
            "Customer ID": [12345.0, None],
            "Country": ["UK ", "UK"],
            "source_sheet": ["A", "A"],
        }
    )


def test_standardize_renames_columns_and_enforces_dtypes(excel_like, monkeypatch):
    monkeypatch.setattr(loader, "get_config_value", fake_config({"data.column_map": COLUMN_MAP}))
    result = loader._standardize(excel_like)

    assert list(result.columns) == [*COLUMN_MAP.values(), "source_sheet"]
    assert result["stock_code"].tolist() == ["M1", "85123A"]
    assert result["invoice"].tolist() == ["1001", "C1002"]
    assert str(result["quantity"].dtype) == "Int64"
    assert pd.isna(result.loc[1, "quantity"])
    assert pd.isna(result.loc[1, "invoice_date"])
    assert str(result["customer_id"].dtype) == "Int64"


def test_standardize_requires_a_column_map(excel_like, monkeypatch):
    monkeypatch.setattr(loader, "get_config_value", fake_config({}))
    with pytest.raises(ValueError, match="column_map"):
        loader._standardize(excel_like)


def test_standardize_reports_missing_excel_columns(excel_like, monkeypatch):
    monkeypatch.setattr(loader, "get_config_value", fake_config({"data.column_map": COLUMN_MAP}))
    with pytest.raises(ValueError, match="not found"):
        loader._standardize(excel_like.drop(columns=["Price"]))


def test_read_excel_sheets_stacks_every_sheet(tmp_path, monkeypatch, excel_like):
    workbook = tmp_path / "retail.xlsx"
    workbook.write_bytes(b"")
    monkeypatch.setattr(loader.pd, "read_excel", lambda *args, **kwargs: excel_like.drop(columns="source_sheet"))

    combined = loader._read_excel_sheets(workbook, ["A", "B"])

    assert len(combined) == 4
    assert list(combined["source_sheet"]) == ["A", "A", "B", "B"]


def test_read_excel_sheets_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raw data file not found"):
        loader._read_excel_sheets(tmp_path / "missing.xlsx", ["A"])


def test_log_summary_handles_unexpected_dates(excel_like, monkeypatch):
    values = {"data.column_map": COLUMN_MAP, "data.expected_date_range": ["2010-01-01", "2010-12-31"]}
    monkeypatch.setattr(loader, "get_config_value", fake_config(values))
    loader._log_summary(loader._standardize(excel_like))  # out-of-range and unparseable dates only warn


@pytest.fixture()
def loader_setup(tmp_path, monkeypatch, excel_like):
    values = {
        "paths.raw_data": "retail.xlsx",
        "paths.interim_dir": "interim",
        "data.sheet_names": ["A"],
        "data.column_map": COLUMN_MAP,
    }
    monkeypatch.setattr(loader, "get_config_value", fake_config(values))
    monkeypatch.setattr(loader, "resolve_path", fake_resolve(tmp_path))
    calls = {"count": 0}

    def fake_reader(path, sheet_names):
        calls["count"] += 1
        return excel_like

    monkeypatch.setattr(loader, "_read_excel_sheets", fake_reader)
    return tmp_path, values, calls


def test_load_transactions_without_cache(loader_setup):
    _, _, calls = loader_setup
    result = loader.load_transactions(use_cache=False)
    assert len(result) == 2 and calls["count"] == 1


def test_load_transactions_requires_sheet_names(loader_setup, monkeypatch):
    _, values, _ = loader_setup
    monkeypatch.setattr(loader, "get_config_value", fake_config({**values, "data.sheet_names": []}))
    with pytest.raises(ValueError, match="sheet_names"):
        loader.load_transactions(use_cache=False)


def test_load_transactions_reuses_a_fresh_cache(loader_setup):
    pytest.importorskip("pyarrow")
    tmp_path, _, calls = loader_setup
    (tmp_path / "retail.xlsx").write_bytes(b"")  # the cache is written after this file, so it is fresh

    first = loader.load_transactions()
    second = loader.load_transactions()

    assert calls["count"] == 1
    assert len(first) == len(second) == 2


def test_force_reload_ignores_the_cache(loader_setup):
    pytest.importorskip("pyarrow")
    tmp_path, _, calls = loader_setup
    (tmp_path / "retail.xlsx").write_bytes(b"")

    loader.load_transactions()
    loader.load_transactions(force_reload=True)

    assert calls["count"] == 2


# ===================================================================== validator
@pytest.fixture()
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "invoice": ["1001", "C1002", "1003"],
            "stock_code": ["M1", "M1", "M2"],
            "description": ["MUG", "MUG", "PEN"],
            "quantity": pd.Series([2, -1, 3], dtype="int64"),
            "invoice_date": pd.to_datetime(["2011-01-05", "2011-02-05", "2011-03-05"]),
            "unit_price": [5.0, 5.0, 2.0],
            "customer_id": pd.Series([1, 1, 2], dtype="int64"),
            "country": "UK",
            "source_sheet": ["A", "A", "B"],
            "is_cancellation": [False, True, False],
            "is_bulk_order": False,
            "is_quantity_outlier": False,
            "line_total": [10.0, -5.0, 6.0],
        }
    )


@pytest.fixture()
def validator_config(monkeypatch):
    values = {"data.max_null_rate.description": 0.1, "data.expected_date_range": ["2011-01-01", "2011-12-31"]}
    monkeypatch.setattr(validator, "get_config_value", fake_config(values))


def test_clean_data_passes_every_check(valid_frame, validator_config):
    report = validate_clean_transactions(valid_frame)
    assert report.ok and not report.failed and len(report.passed) >= 12


def _break_price(frame):
    frame.loc[0, "unit_price"] = 0.0


def _break_customer(frame):
    frame.loc[0, "customer_id"] = 0


def _sale_negative(frame):
    frame.loc[0, "quantity"] = -2


def _return_positive(frame):
    frame.loc[1, "quantity"] = 1


def _duplicate_across_sheets(frame):
    frame.loc[1, "source_sheet"] = "B"
    frame.loc[0, "invoice"] = "C1002"
    frame.loc[0, "source_sheet"] = "A"


def _date_out_of_range(frame):
    frame.loc[0, "invoice_date"] = pd.Timestamp("2015-01-01")


def _description_missing(frame):
    frame.loc[:1, "description"] = None


def _key_missing(frame):
    frame["stock_code"] = frame["stock_code"].astype(object)
    frame.loc[0, "stock_code"] = None


@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        (_break_price, "unit_price > 0"),
        (_break_customer, "customer_id > 0"),
        (_sale_negative, "sales have positive quantity"),
        (_return_positive, "cancellations have negative quantity"),
        (_duplicate_across_sheets, "more than one sheet"),
        (_date_out_of_range, "inside expected range"),
        (_description_missing, "description within allowed rate"),
        (_key_missing, "no missing values in key columns"),
    ],
)
def test_each_broken_rule_is_reported(valid_frame, validator_config, break_it, expected):
    break_it(valid_frame)
    report = validate_clean_transactions(valid_frame, raise_on_error=False)

    assert not report.ok
    assert any(expected in message for message in report.failed)


def test_failed_checks_raise_by_default(valid_frame, validator_config):
    valid_frame.loc[0, "unit_price"] = -1.0
    with pytest.raises(DataValidationError, match="Validation failed"):
        validate_clean_transactions(valid_frame)


def test_missing_columns_stop_validation_early(valid_frame, validator_config):
    report = validate_clean_transactions(valid_frame.drop(columns=["line_total"]), raise_on_error=False)
    assert not report.ok
    assert len(report.passed) + len(report.failed) == 2  # schema and empty-check only


def test_empty_data_fails_validation(valid_frame, validator_config):
    report = validate_clean_transactions(valid_frame.iloc[0:0], raise_on_error=False)
    assert any("not empty" in message for message in report.failed)


def test_validation_report_records_passes_and_failures():
    report = ValidationReport()
    report.check("good", True)
    report.check("bad", False, "because")
    assert report.passed == ["good"] and report.failed == ["bad - because"] and not report.ok


def test_load_clean_transactions_reads_the_interim_file(valid_frame, tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(validator, "resolve_path", fake_resolve(tmp_path))
    monkeypatch.setattr(validator, "get_config_value", fake_config({}))

    with pytest.raises(FileNotFoundError, match="src.data.cleaner"):
        validator.load_clean_transactions()

    valid_frame.to_parquet(tmp_path / "transactions_clean.parquet", index=False)
    assert len(validator.load_clean_transactions()) == 3