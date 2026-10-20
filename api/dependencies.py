"""FastAPI dependencies."""
from __future__ import annotations

from fastapi import Request

from api.services import PredictionService


def get_service(request: Request) -> PredictionService:
    """Return the prediction service created at application start-up."""
    return request.app.state.service