"""Service health and model information."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.prediction import HealthResponse
from api.services import PredictionService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(service: PredictionService = Depends(get_service)) -> dict:
    """Confirm the API is running and list the loaded models and required features."""
    return service.health()