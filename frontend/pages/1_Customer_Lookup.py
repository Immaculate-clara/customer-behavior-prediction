"""Look up one customer: churn risk, expected revenue, segment and the reasons behind the score."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
import charts
import loaders
import ui
from insights import money, pretty_feature, recommend

ui.setup_page("Customer Lookup", "🔎")

# Start with the customer who most needs attention so the page is never empty.
top = loaders.portfolio_customers("high", None, "priority", 1)["customers"]
default_id = int(top[0]["customer_id"]) if top else 0

controls = st.columns([2, 2, 3])
customer_id = controls[0].number_input("Customer ID", min_value=0, value=default_id, step=1)
top_n = controls[1].slider("Reasons to show", 3, 15, 8)

try:
    profile = loaders.customer_profile(int(customer_id))
except api_client.ApiError as exc:
    st.warning(f"{exc}")
    st.caption("Tip: use an ID from the 'Who to contact first' list on the Overview page.")
    st.stop()

churn, clv, segment = profile["churn"], profile["clv"], profile["segment"]
risk = churn["risk_level"]

left, middle, right = st.columns([3, 3, 4])
with left:
    ui.show_figure(charts.probability_gauge(churn["churn_probability"]))
with middle:
    st.metric("Risk level", risk.title())
    st.metric("Expected revenue, next 90 days", money(clv["predicted_revenue"]))
    st.metric("Segment", segment["segment_name"])
with right:
    st.markdown("#### Suggested action")
    st.info(recommend(risk, segment["segment_name"]))
    st.caption(
        f"The model flags a customer as a likely churner above a probability of {churn['threshold']:.0%} "
        f"(model: {churn['model']})."
    )

st.subheader("Why this score?")
try:
    explanation = loaders.explain_churn(int(customer_id), int(top_n))
except api_client.ApiError as exc:
    st.error(str(exc))
    st.stop()

ui.show_figure(charts.shap_chart(explanation["top_features"]))
table = pd.DataFrame(explanation["top_features"])
ui.show_table(
    pd.DataFrame(
        {
            "Feature": table["feature"].map(pretty_feature),
            "Customer's value": table["value"].round(2),
            "Effect": table["effect"].str.capitalize(),
            "Strength (SHAP)": table["shap_value"].round(3),
        }
    )
)
st.caption("SHAP values show how much each feature moved this customer's churn score away from the average customer.")