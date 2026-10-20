"""Overview page of the Customer Behavior Prediction dashboard.

Run from the project root:  streamlit run frontend/app.py
(the API must be running: uvicorn api.main:app --port 8000)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
import charts
import loaders
import ui
from insights import money

ui.setup_page("Customer Overview", "📊")

summary = loaders.portfolio_summary()

# ---- Key numbers ---------------------------------------------------------------------------
columns = st.columns(5)
columns[0].metric("Customers scored", f"{summary['customers']:,}")
columns[1].metric("Average churn risk", f"{summary['avg_churn_probability']:.0%}")
columns[2].metric(
    "High-risk customers",
    f"{summary['high_risk_customers']:,}",
    f"{summary['high_risk_share']:.0%} of customers",
    delta_color="off",
)
columns[3].metric("Expected revenue, next 90 days", money(summary["expected_revenue"]))
columns[4].metric("Past spend of high-risk customers", money(summary["high_risk_historical_spend"]))

# ---- Risk charts ---------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    ui.show_figure(charts.risk_distribution_chart(summary["risk_distribution"]))
with right:
    ui.show_figure(charts.churn_histogram_chart(summary["churn_probability_histogram"]))

left, right = st.columns(2)
with left:
    ui.show_figure(charts.segment_churn_chart(summary["segments"]))
with right:
    ui.show_figure(charts.segment_revenue_chart(summary["segments"]))

# ---- Retention action list ------------------------------------------------------------------
st.subheader("Who to contact first")
st.caption(
    "Customers ranked by churn probability x past spend, so likely leavers who matter most to revenue come first."
)

SORT_OPTIONS = {
    "Retention priority": "priority",
    "Churn probability": "churn_probability",
    "Expected revenue": "predicted_revenue",
    "Past spend": "monetary_total",
}
filters = st.columns(4)
risk_choice = filters[0].selectbox("Risk level", ["high", "medium", "low", "all"], index=0)
segment_names = ["All segments"] + [item["segment"] for item in summary["segments"]]
segment_choice = filters[1].selectbox("Segment", segment_names)
sort_choice = filters[2].selectbox("Rank by", list(SORT_OPTIONS))
limit = filters[3].slider("Customers to show", 10, 200, 50, step=10)

try:
    result = loaders.portfolio_customers(
        None if risk_choice == "all" else risk_choice,
        None if segment_choice == "All segments" else segment_choice,
        SORT_OPTIONS[sort_choice],
        limit,
    )
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

if not result["customers"]:
    st.info("No customers match these filters.")
else:
    table = pd.DataFrame(result["customers"])
    display = pd.DataFrame(
        {
            "Customer ID": table["customer_id"],
            "Churn probability (%)": table["churn_probability"] * 100,
            "Risk": table["risk_level"].str.title(),
            "Segment": table["segment_name"],
            "Expected revenue (90 days)": table["predicted_revenue"],
            "Past spend": table["monetary_total"],
            "Days since last order": table["recency_days"].round(0),
            "Orders": table["frequency"].round(0),
        }
    )
    st.caption(f"Showing {result['returned']:,} of {result['total']:,} matching customers")
    ui.show_table(
        display,
        column_config={
            "Churn probability (%)": st.column_config.ProgressColumn(
                "Churn probability", min_value=0, max_value=100, format="%.0f%%"
            ),
            "Expected revenue (90 days)": st.column_config.NumberColumn(format="£%.0f"),
            "Past spend": st.column_config.NumberColumn(format="£%.0f"),
        },
    )
    st.download_button(
        "Download this list (CSV)", display.to_csv(index=False).encode("utf-8"), "retention_list.csv", "text/csv"
    )