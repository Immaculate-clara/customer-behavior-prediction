"""Prediction endpoints: churn, CLV, segment and batch scoring."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.prediction import (
    BatchRequest,
    BatchResponse,
    ChurnPrediction,
    ClvPrediction,
    CustomerInput,
    SegmentPrediction,
)
from api.services import PredictionService

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/churn", response_model=ChurnPrediction)
def predict_churn(payload: CustomerInput, service: PredictionService = Depends(get_service)) -> dict:
    """Probability that the customer buys nothing in the next 90 days, plus a risk level."""
    X, ids = service.resolve(payload.customer_id, payload.features)
    return service.churn(X, ids)[0]


@router.post("/clv", response_model=ClvPrediction)
def predict_clv(payload: CustomerInput, service: PredictionService = Depends(get_service)) -> dict:
    """Expected revenue from the customer over the next 90 days."""
    X, ids = service.resolve(payload.customer_id, payload.features)
    return service.clv(X, ids)[0]


@router.post("/segment", response_model=SegmentPrediction)
def predict_segment(payload: CustomerInput, service: PredictionService = Depends(get_service)) -> dict:
    """Customer segment (stored value for known customers, K-Means assignment for new ones)."""
    X, ids = service.resolve(payload.customer_id, payload.features)
    return service.segment(X, ids)[0]


@router.post("/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, service: PredictionService = Depends(get_service)) -> dict:
    """Churn, CLV and segment for up to 1000 known customers in one call."""
    X, ids = service.resolve_many(payload.customer_ids)
    results = service.profiles(X, ids)
    return {"count": len(results), "results": results}