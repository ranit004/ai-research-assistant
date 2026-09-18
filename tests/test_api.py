"""Tests for POST /ingest/document and POST /retrieve.

The Qdrant client and embedder are mocked so these tests run without any
external services.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── /ingest/document ───────────────────────────────────────────────────────────


def test_ingest_plain_text_returns_200(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "validate_upload", lambda data, name: "text/plain")
    monkeypatch.setattr(ingest_mod, "extract_text", lambda data, mime: "Sample document text " * 20)
    monkeypatch.setattr(ingest_mod, "clean_text", lambda t: t)
    monkeypatch.setattr(ingest_mod, "get_embeddings", lambda texts: [[0.1] * 3 for _ in texts])
    monkeypatch.setattr(ingest_mod, "ensure_collection", lambda: None)
    monkeypatch.setattr(ingest_mod, "upsert_chunks", lambda chunks, vectors: len(chunks))

    response = client.post(
        "/ingest/document",
        files={"file": ("sample.txt", io.BytesIO(b"Sample document text " * 20), "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "sample"
    assert body["chunk_count"] > 0
    assert body["document_id"]


def test_ingest_rejects_oversized_file(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "validate_upload", lambda data, name: (_ for _ in ()).throw(ValueError("exceeds")))

    response = client.post(
        "/ingest/document",
        files={"file": ("big.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 422


def test_ingest_rejects_disallowed_type(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.ingest as ingest_mod

    monkeypatch.setattr(
        ingest_mod,
        "validate_upload",
        lambda data, name: (_ for _ in ()).throw(ValueError("Unsupported file type")),
    )

    response = client.post(
        "/ingest/document",
        files={"file": ("img.png", io.BytesIO(b"\x89PNG"), "image/png")},
    )
    assert response.status_code == 422


def test_ingest_returns_503_when_no_openai_key(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "validate_upload", lambda data, name: "text/plain")
    monkeypatch.setattr(ingest_mod, "extract_text", lambda data, mime: "text " * 50)
    monkeypatch.setattr(ingest_mod, "clean_text", lambda t: t)
    monkeypatch.setattr(
        ingest_mod,
        "get_embeddings",
        lambda texts: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY is not configured")),
    )

    response = client.post(
        "/ingest/document",
        files={"file": ("doc.txt", io.BytesIO(b"text " * 50), "text/plain")},
    )
    assert response.status_code == 503


# ── /retrieve ─────────────────────────────────────────────────────────────────


def _make_hit(chunk_id: str, score: float) -> MagicMock:
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = score
    hit.payload = {
        "document_id": "doc-1",
        "chunk_index": 0,
        "title": "Test",
        "section": "",
        "source": "test.txt",
        "text": "Some retrieved text.",
    }
    return hit


def test_retrieve_returns_results(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.retrieve as retrieve_mod

    monkeypatch.setattr(retrieve_mod, "get_embeddings", lambda texts: [[0.1] * 3])
    monkeypatch.setattr(
        retrieve_mod,
        "similarity_search",
        lambda query_vector, top_k, score_threshold: [_make_hit("c1", 0.9)],
    )

    response = client.post("/retrieve", json={"query": "what is the document about?"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["score"] == pytest.approx(0.9)


def test_retrieve_returns_503_when_no_key(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.retrieve as retrieve_mod

    monkeypatch.setattr(
        retrieve_mod,
        "get_embeddings",
        lambda texts: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY is not configured")),
    )

    response = client.post("/retrieve", json={"query": "test"})
    assert response.status_code == 503


def test_retrieve_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/retrieve", json={"query": ""})
    assert response.status_code == 422


def test_retrieve_rejects_top_k_out_of_range(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.retrieve as retrieve_mod

    monkeypatch.setattr(retrieve_mod, "get_embeddings", lambda texts: [[0.1] * 3])

    response = client.post("/retrieve", json={"query": "test", "top_k": 0})
    assert response.status_code == 422
