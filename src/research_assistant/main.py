"""FastAPI application entry point.

Run locally with:
    uvicorn research_assistant.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from research_assistant.api.health import router as health_router
from research_assistant.api.ingest import router as ingest_router
from research_assistant.api.research import router as research_router
from research_assistant.api.retrieve import router as retrieve_router
from research_assistant.config import settings
from research_assistant.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting %s (environment=%s)", settings.app_name, settings.environment)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    setup_logging(settings.log_level)
    application = FastAPI(
        title=settings.app_name,
        description="Backend-only multi-step research assistant API.",
        version=__import__("research_assistant").__version__,
        lifespan=lifespan,
        debug=settings.debug,
    )

    # Generic 500 handler: never expose stack traces through the API.
    @application.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    application.include_router(health_router)
    application.include_router(ingest_router)
    application.include_router(retrieve_router)
    application.include_router(research_router)
    return application


app = create_app()
