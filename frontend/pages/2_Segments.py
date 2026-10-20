"""Customer segments: size, risk, revenue and typical behaviour."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import loaders
import ui

ui.setup_page("Customer Segments", "🧩")

summary = loaders.portfolio_summary()
profiles = loaders.segments()
risk_by_segment = {item["segment"]: item for item in summary["segments"]}

left, right = st.columns(2)
with left:
    ui.show_figure(charts.segment_size_chart(summary["segments"]))
with right:
    ui.show_figure(charts.segment_churn_chart(summary["segments"]))

st.subheader("Segment overview")
overview = pd.DataFrame(
    [
        {
            "Segment": item["segment"],
            "Customers": item["customers"],
            "Share of customers (%)": item["share_pct"],
            "Average churn risk (%)": risk_by_segment[item["segment"]]["avg_churn_probability"] * 100,
            "High-risk customers (%)": risk_by_segment[item["segment"]]["high_risk_share"] * 100,
            "Expected revenue (90 days)": risk_by_segment[item["segment"]]["expected_revenue"],
            "Total past spend": risk_by_segment[item["segment"]]["historical_spend"],
        }
        for item in profiles
        if item["segment"] in risk_by_segment
    ]
).sort_values("Expected revenue (90 days)", ascending=False)
ui.show_table(
    overview,
    column_config={
        "Share of customers (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Average churn risk (%)": st.column_config.NumberColumn(format="%.0f%%"),
        "High-risk customers (%)": st.column_config.NumberColumn(format="%.0f%%"),
        "Expected revenue (90 days)": st.column_config.NumberColumn(format="£%.0f"),
        "Total past spend": st.column_config.NumberColumn(format="£%.0f"),
    },
)

st.subheader("How the segments behave")
HEATMAP_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_basket_size",
    "tenure_days",
    "interpurchase_mean_days",
    "discount_line_share",
    "seasonal_order_share",
    "return_rate",
]
ui.show_figure(charts.segment_heatmap(profiles, HEATMAP_FEATURES))

st.subheader("Customers in a segment")
chosen = st.selectbox("Segment", [item["segment"] for item in profiles])
members = loaders.portfolio_customers(None, chosen, "priority", 25)
if members["customers"]:
    table = pd.DataFrame(members["customers"])
    st.caption(f"Top 25 of {members['total']:,} customers by retention priority")
    ui.show_table(
        pd.DataFrame(
            {
                "Customer ID": table["customer_id"],
                "Churn probability (%)": table["churn_probability"] * 100,
                "Expected revenue (90 days)": table["predicted_revenue"],
                "Past spend": table["monetary_total"],
                "Days since last order": table["recency_days"].round(0),
            }
        ),
        column_config={
            "Churn probability (%)": st.column_config.ProgressColumn(
                "Churn probability", min_value=0, max_value=100, format="%.0f%%"
            ),
            "Expected revenue (90 days)": st.column_config.NumberColumn(format="£%.0f"),
            "Past spend": st.column_config.NumberColumn(format="£%.0f"),
        },
    )