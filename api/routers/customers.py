"""Customer profiles and segment summaries."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.prediction import CustomerProfile, SegmentSummary
from api.services import PredictionService

router = APIRouter(tags=["customers"])


@router.get("/customers/{customer_id}", response_model=CustomerProfile)
def get_customer(customer_id: int, service: PredictionService = Depends(get_service)) -> dict:
    """Churn risk, expected revenue and segment of one known customer."""
    X, ids = service.resolve_many([customer_id])
    return service.profiles(X, ids)[0]


@router.get("/segments", response_model=list[SegmentSummary])
def list_segments(service: PredictionService = Depends(get_service)) -> list[dict]:
    """Size, churn rate, revenue and average behaviour of every segment."""
    return service.segment_summaries()