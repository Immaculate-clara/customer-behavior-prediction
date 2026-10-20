"""Explanation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.prediction import ExplainRequest, ExplanationResponse
from api.services import PredictionService

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("/churn", response_model=ExplanationResponse)
def explain_churn(payload: ExplainRequest, service: PredictionService = Depends(get_service)) -> dict:
    """Features that push this customer's churn score up or down (SHAP values)."""
    X, ids = service.resolve(payload.customer_id, payload.features)
    return service.explain(X, ids, payload.top_n)[0]