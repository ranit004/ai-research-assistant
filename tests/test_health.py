"""Tests for GET /health."""

from fastapi.testclient import TestClient

from research_assistant.config import settings


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_identity(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["app"] == settings.app_name
    assert body["environment"] == settings.environment
    assert body["version"]
