"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from research_assistant.main import app


@pytest.fixture()
def client() -> TestClient:
    """Test client that also runs the application lifespan."""
    with TestClient(app) as test_client:
        yield test_client
