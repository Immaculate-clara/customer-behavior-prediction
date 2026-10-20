"""Shared Streamlit helpers: page set-up, API status and version-safe widgets."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
import loaders


def setup_page(title: str, icon: str) -> dict:
    """Configure the page, show the header and API status; stop with a clear message if the API is down."""
    st.set_page_config(page_title=f"{title} | Customer Behavior Prediction", page_icon=icon, layout="wide")
    st.markdown(
        "<style>div[data-testid='stMetricValue']{font-size:1.7rem;} .block-container{padding-top:2rem;}</style>",
        unsafe_allow_html=True,
    )
    try:
        status = loaders.health()
    except api_client.ApiError as exc:
        st.title(f"{icon} {title}")
        st.error(str(exc))
        st.stop()
    with st.sidebar:
        st.success("API connected")
        st.caption(
            f"Churn model: **{status['churn_model']}**  \n"
            f"CLV model: **{status['clv_model']}**  \n"
            f"Customers: **{status['customers_available']:,}**"
        )
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()
    st.title(f"{icon} {title}")
    return status


def show_figure(figure) -> None:
    """Display a Plotly figure at full width (works across Streamlit versions)."""
    try:
        st.plotly_chart(figure, width="stretch")
    except TypeError:
        st.plotly_chart(figure, use_container_width=True)


def show_table(frame: pd.DataFrame, **kwargs) -> None:
    """Display a dataframe at full width (works across Streamlit versions)."""
    try:
        st.dataframe(frame, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)