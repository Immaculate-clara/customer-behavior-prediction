"""Request and response models for the prediction API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RiskLevel = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class CustomerInput(BaseModel):
    """Identify a customer either by ``customer_id`` or by a full ``features`` dictionary."""

    customer_id: int | None = Field(default=None, ge=0, description="ID of a customer already in the feature table")
    features: dict[str, float] | None = Field(
        default=None, description="All model features for a new customer (see GET /health for the list)"
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "CustomerInput":
        if (self.customer_id is None) == (self.features is None):
            raise ValueError("Provide exactly one of 'customer_id' or 'features'")
        return self


class ExplainRequest(CustomerInput):
    top_n: int = Field(default=5, ge=1, le=15, description="Number of features to return")


class BatchRequest(BaseModel):
    customer_ids: list[int] = Field(min_length=1, max_length=1000)


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class ChurnPrediction(BaseModel):
    customer_id: int | None
    churn_probability: float = Field(ge=0, le=1)
    predicted_churn: bool
    risk_level: RiskLevel
    threshold: float
    model: str


class ClvPrediction(BaseModel):
    customer_id: int | None
    predicted_revenue: float = Field(description="Expected revenue over the churn horizon (currency units)")
    model: str


class SegmentPrediction(BaseModel):
    customer_id: int | None
    segment_id: int
    segment_name: str


class CustomerProfile(BaseModel):
    customer_id: int | None
    churn: ChurnPrediction
    clv: ClvPrediction
    segment: SegmentPrediction


class BatchResponse(BaseModel):
    count: int
    results: list[CustomerProfile]


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    effect: str


class ExplanationResponse(BaseModel):
    customer_id: int | None
    churn_probability: float
    top_features: list[FeatureContribution]


class SegmentSummary(BaseModel):
    segment_id: int
    segment: str
    customers: int
    share_pct: float
    churn_rate: float | None = None
    avg_future_revenue: float | None = None
    averages: dict[str, float] = Field(default_factory=dict, description="Average feature values in the segment")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    churn_model: str
    clv_model: str
    customers_available: int
    feature_columns: list[str]