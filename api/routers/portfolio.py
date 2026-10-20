"""Dashboard endpoints: portfolio summary, customer action list and model comparison."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_portfolio
from api.portfolio import Portfolio
from api.schemas.portfolio import CustomerList, ModelComparison, PortfolioSummary
from api.schemas.prediction import RiskLevel
from src.utils.config import get_config_value, resolve_path

router = APIRouter(tags=["portfolio"])

SortBy = Literal["priority", "churn_probability", "predicted_revenue", "monetary_total"]
COMPARISON_FILES = {
    "churn": "churn_model_comparison.csv",
    "deep_learning": "deep_model_comparison.csv",
    "clv": "clv_model_comparison.csv",
}


@router.get("/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary(portfolio: Portfolio = Depends(get_portfolio)) -> dict:
    """Risk distribution, expected revenue and per-segment totals for all customers."""
    return portfolio.summary()


@router.get("/portfolio/customers", response_model=CustomerList)
def portfolio_customers(
    risk_level: RiskLevel | None = None,
    segment: str | None = Query(default=None, description="Segment name, e.g. Champions"),
    sort_by: SortBy = "priority",
    limit: int = Query(default=50, ge=1, le=500),
    portfolio: Portfolio = Depends(get_portfolio),
) -> dict:
    """Customers ranked for retention action (default: churn probability x past spend)."""
    return portfolio.customers(risk_level=risk_level, segment=segment, sort_by=sort_by, limit=limit)


@router.get("/models/comparison", response_model=ModelComparison)
def model_comparison() -> dict:
    """Test-set results of every trained model (read from the saved comparison tables)."""
    artifacts = resolve_path(get_config_value("paths.artifacts_dir", "models_artifacts"))
    result = {}
    for key, filename in COMPARISON_FILES.items():
        path = artifacts / filename
        if path.exists():
            frame = pd.read_csv(path).astype(object).where(lambda df: df.notna(), None)
            result[key] = frame.to_dict(orient="records")
        else:
            result[key] = []
    return result