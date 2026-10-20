"""Response models for the dashboard (portfolio and model-comparison) endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from api.schemas.prediction import RiskLevel


class HistogramBin(BaseModel):
    start: float
    end: float
    count: int


class SegmentRisk(BaseModel):
    segment_id: int
    segment: str
    customers: int
    avg_churn_probability: float
    high_risk_share: float
    expected_revenue: float
    historical_spend: float


class PortfolioSummary(BaseModel):
    customers: int
    avg_churn_probability: float
    high_risk_customers: int
    high_risk_share: float
    expected_revenue: float
    high_risk_historical_spend: float
    risk_distribution: dict[str, int]
    churn_probability_histogram: list[HistogramBin]
    segments: list[SegmentRisk]


class PortfolioCustomer(BaseModel):
    customer_id: int
    churn_probability: float
    risk_level: RiskLevel
    predicted_churn: bool
    predicted_revenue: float
    segment_id: int
    segment_name: str
    recency_days: float
    frequency: float
    monetary_total: float
    priority_score: float


class CustomerList(BaseModel):
    total: int
    returned: int
    customers: list[PortfolioCustomer]


class ModelComparison(BaseModel):
    churn: list[dict[str, Any]]
    deep_learning: list[dict[str, Any]]
    clv: list[dict[str, Any]]