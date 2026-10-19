"""Unit tests for feature engineering, with explicit point-in-time leakage checks."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.affinity import (
    CATEGORIES,
    build_affinity_features,
    build_next_category_labels,
    categorize_descriptions,
)
from src.features.build_features import build_customer_features, build_labels

CUTOFF = pd.Timestamp("2011-09-09")
HORIZON = 90


def _transactions(rows: list[tuple]) -> pd.DataFrame:
    """Build a transactions frame from (invoice, stock_code, description, qty, date, price, customer)."""
    df = pd.DataFrame(
        rows,
        columns=["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id"],
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["is_cancellation"] = df["invoice"].str.startswith("C")
    df["line_total"] = df["quantity"] * df["unit_price"]
    return df


@pytest.fixture()
def transactions() -> pd.DataFrame:
    return _transactions(
        [
            # Customer 1: two orders and one return before the cutoff, one order after.
            ("1001", "M1", "RED MUG", 2, "2011-08-01 10:00", 10.0, 1),
            ("1002", "X1", "CHRISTMAS STAR", 1, "2011-09-01 10:00", 30.0, 1),
            ("C1003", "M1", "RED MUG", -1, "2011-09-05 10:00", 10.0, 1),
            ("1004", "M1", "RED MUG", 5, "2011-10-01 10:00", 1.0, 1),
            # Customer 2: one order before the cutoff, nothing after (churned).
            ("1005", "M1", "RED MUG", 1, "2011-07-01 00:00", 100.0, 2),
            # Customer 3: first ever order exactly AT the cutoff -> no history.
            ("1006", "M1", "RED MUG", 1, "2011-09-09 00:00", 5.0, 3),
            # Customer 4: order just before the cutoff plus one exactly at the cutoff.
            ("1007", "M1", "RED MUG", 1, "2011-09-08 23:59", 7.0, 4),
            ("1008", "M1", "RED MUG", 1, "2011-09-09 00:00", 9.0, 4),
            # Late row so the dataset covers the full 90-day horizon.
            ("1009", "M1", "RED MUG", 1, "2011-12-08 10:00", 1.0, 3),
        ]
    )


# ----------------------------------------------------------------------------
# Hand-computed values
# ----------------------------------------------------------------------------
def test_core_features_match_hand_calculation(transactions):
    features = build_customer_features(transactions, CUTOFF)
    customer = features.loc[1]

    assert customer["frequency"] == 2
    assert customer["monetary_total"] == pytest.approx(50.0)
    assert customer["monetary_avg"] == pytest.approx(25.0)
    assert customer["recency_days"] == pytest.approx(7.583333, abs=1e-4)  # Sep 1 10:00 -> Sep 9 00:00
    assert customer["n_returns"] == 1
    assert customer["return_rate"] == pytest.approx(0.2)
    assert customer["historical_clv"] == pytest.approx(40.0)
    assert features.loc[2, "recency_days"] == pytest.approx(70.0)


def test_labels_match_hand_calculation(transactions):
    features = build_customer_features(transactions, CUTOFF)
    labels = build_labels(transactions, features.index, CUTOFF, HORIZON)

    assert labels.loc[1, "churn"] == 0
    assert labels.loc[1, "future_revenue"] == pytest.approx(5.0)
    assert labels.loc[2, "churn"] == 1
    assert labels.loc[2, "future_revenue"] == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# Leakage tests (PRD Section 7)
# ----------------------------------------------------------------------------
def test_features_ignore_all_data_on_or_after_cutoff(transactions):
    baseline = build_customer_features(transactions, CUTOFF)

    altered = transactions.copy()
    future = altered["invoice_date"] >= CUTOFF
    altered.loc[future, "unit_price"] *= 50
    altered.loc[future, "line_total"] *= 50
    extra = altered[future].copy()
    extra["invoice"] = "F" + extra["invoice"]
    altered = pd.concat([altered, extra], ignore_index=True)

    pd.testing.assert_frame_equal(
        baseline.sort_index(), build_customer_features(altered, CUTOFF).sort_index()
    )


def test_cutoff_boundary_is_strictly_before(transactions):
    features = build_customer_features(transactions, CUTOFF)
    labels = build_labels(transactions, features.index, CUTOFF, HORIZON)

    assert 3 not in features.index  # only order is exactly at the cutoff
    assert features.loc[4, "frequency"] == 1  # the cutoff-time order is excluded
    assert labels.loc[4, "churn"] == 0  # ...but counts as the future purchase
    assert labels.loc[4, "future_revenue"] == pytest.approx(9.0)


def test_labels_ignore_history(transactions):
    features = build_customer_features(transactions, CUTOFF)
    baseline = build_labels(transactions, features.index, CUTOFF, HORIZON)

    altered = transactions.copy()
    past = altered["invoice_date"] < CUTOFF
    altered.loc[past, "line_total"] *= 99
    after = build_labels(altered, features.index, CUTOFF, HORIZON)

    pd.testing.assert_frame_equal(baseline, after)


def test_labels_reject_incomplete_horizon(transactions):
    early = transactions[transactions["invoice_date"] < "2011-11-01"]
    with pytest.raises(ValueError, match="horizon"):
        build_labels(early, pd.Index([1, 2]), CUTOFF, HORIZON)


def test_features_have_no_missing_values(transactions):
    features = build_customer_features(transactions, CUTOFF)
    assert not features.isna().any().any()


# ----------------------------------------------------------------------------
# Category affinity
# ----------------------------------------------------------------------------
def test_affinity_shares_sum_to_one_and_match_spend(transactions):
    affinity = build_affinity_features(transactions, CUTOFF)
    share_columns = [col for col in affinity.columns if col.startswith("affinity_")]

    assert len(share_columns) == len(CATEGORIES)
    assert affinity[share_columns].sum(axis=1).tolist() == pytest.approx([1.0] * len(affinity))
    assert affinity.loc[1, "affinity_kitchen_dining"] == pytest.approx(0.4)
    assert affinity.loc[1, "affinity_seasonal_christmas"] == pytest.approx(0.6)
    assert affinity.loc[1, "top_category"] == "Seasonal & Christmas"


def test_affinity_ignores_future_data(transactions):
    baseline = build_affinity_features(transactions, CUTOFF)

    altered = transactions.copy()
    future = altered["invoice_date"] >= CUTOFF
    altered.loc[future, "stock_code"] = "X1"
    altered.loc[future, "line_total"] *= 25

    pd.testing.assert_frame_equal(
        baseline.sort_index(), build_affinity_features(altered, CUTOFF).sort_index()
    )


def test_next_category_label_is_missing_only_for_churned_customers(transactions):
    features = build_customer_features(transactions, CUTOFF)
    labels = build_labels(transactions, features.index, CUTOFF, HORIZON)
    next_category = build_next_category_labels(transactions, features.index, CUTOFF, HORIZON)

    assert next_category.loc[1] == "Kitchen & Dining"
    assert ((labels["churn"] == 1) == next_category.isna()).all()


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("WHITE HANGING HEART T-LIGHT HOLDER", "Candles & Lighting"),
        ("PAPER CHAIN KIT 50'S CHRISTMAS", "Seasonal & Christmas"),
        ("JUMBO BAG RED RETROSPOT", "Bags & Storage"),
        ("SET/6 RED SPOTTY PAPER CUPS", "Kitchen & Dining"),
        ("PINK BUTTERFLY NECKLACE", "Jewellery & Accessories"),
        ("SOMETHING COMPLETELY UNKNOWN", "Other"),
    ],
)
def test_description_categorisation(description, expected):
    assert categorize_descriptions(pd.Series([description])).iloc[0] == expected