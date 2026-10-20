"""FastAPI dependencies."""
from __future__ import annotations

from fastapi import Request

from api.portfolio import Portfolio
from api.services import PredictionService


def get_service(request: Request) -> PredictionService:
    """Return the prediction service created at application start-up."""
    return request.app.state.service


def get_portfolio(request: Request) -> Portfolio:
    """Return the scored customer table created at application start-up."""
    return request.app.state.portfolio