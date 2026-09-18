"""/health endpoint: liveness and basic service identity."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from research_assistant import __version__
from research_assistant.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    environment: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service liveness and identity."""
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
        version=__version__,
    )
