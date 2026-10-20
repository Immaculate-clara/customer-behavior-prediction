"""Cached API calls for the dashboard (results are reused for five minutes)."""
from __future__ import annotations

from typing import Any

import streamlit as st

import api_client

CACHE_SECONDS = 300


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def health() -> dict[str, Any]:
    return api_client.health()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Loading portfolio...")
def portfolio_summary() -> dict[str, Any]:
    return api_client.portfolio_summary()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def portfolio_customers(risk_level: str | None, segment: str | None, sort_by: str, limit: int) -> dict[str, Any]:
    return api_client.portfolio_customers(risk_level, segment, sort_by, limit)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def segments() -> list[dict[str, Any]]:
    return api_client.segments()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def customer_profile(customer_id: int) -> dict[str, Any]:
    return api_client.customer_profile(customer_id)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Explaining the prediction...")
def explain_churn(customer_id: int, top_n: int) -> dict[str, Any]:
    return api_client.explain_churn(customer_id, top_n)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def model_comparison() -> dict[str, list[dict[str, Any]]]:
    return api_client.model_comparison()