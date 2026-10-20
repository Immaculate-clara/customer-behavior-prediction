"""FastAPI application for the Customer Behavior Prediction Platform.

Run from the project root:  uvicorn api.main:app --reload --port 8000
Interactive documentation: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.portfolio import Portfolio
from api.routers import customers, explain, health, portfolio, predict
from api.services import CustomerNotFound, InvalidFeatures, PredictionService
from src.utils.config import get_config_value
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model once at start-up and score all customers for the dashboard."""
    app.state.service = PredictionService()
    app.state.portfolio = Portfolio(app.state.service)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=str(get_config_value("api.title", "Customer Behavior Prediction API")),
        description="Churn risk, customer lifetime value, segmentation and explanations for retail customers.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_config_value("api.cors_origins", ["http://localhost:8501", "http://127.0.0.1:8501"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(CustomerNotFound)
    async def _not_found(_: Request, exc: CustomerNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidFeatures)
    async def _invalid_features(_: Request, exc: InvalidFeatures) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    for module in (health, predict, customers, explain, portfolio):
        app.include_router(module.router)
    return app


app = create_app()