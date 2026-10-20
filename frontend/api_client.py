"""Thin HTTP client for the prediction API (no Streamlit code, so it is easy to test)."""
from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 60  # the first explanation request builds the SHAP explainer


class ApiError(RuntimeError):
    """Raised when the API is unreachable or returns an error."""


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{API_URL}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Cannot reach the API at {API_URL}. Start it with: uvicorn api.main:app --port 8000"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"The API at {API_URL} did not answer in {TIMEOUT_SECONDS} seconds") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"API error {response.status_code}: {detail}")
    return response.json()


def health() -> dict[str, Any]:
    return _request("GET", "/health")


def portfolio_summary() -> dict[str, Any]:
    return _request("GET", "/portfolio/summary")


def portfolio_customers(
    risk_level: str | None = None,
    segment: str | None = None,
    sort_by: str = "priority",
    limit: int = 50,
) -> dict[str, Any]:
    params: dict[str, Any] = {"sort_by": sort_by, "limit": limit}
    if risk_level:
        params["risk_level"] = risk_level
    if segment:
        params["segment"] = segment
    return _request("GET", "/portfolio/customers", params=params)


def segments() -> list[dict[str, Any]]:
    return _request("GET", "/segments")


def customer_profile(customer_id: int) -> dict[str, Any]:
    return _request("GET", f"/customers/{int(customer_id)}")


def explain_churn(customer_id: int, top_n: int = 8) -> dict[str, Any]:
    return _request("POST", "/explain/churn", json={"customer_id": int(customer_id), "top_n": int(top_n)})


def model_comparison() -> dict[str, list[dict[str, Any]]]:
    return _request("GET", "/models/comparison")