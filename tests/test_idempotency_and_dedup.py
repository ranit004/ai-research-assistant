"""Comprehensive test suite for document ingestion idempotency, deterministic fingerprinting,
title extraction fixes, and retrieval deduplication.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.models import QueryResponse, ScoredPoint

from research_assistant.ingestion.chunker import (
    chunk_text,
    compute_document_hash,
    generate_chunk_id,
    generate_document_id,
)
from research_assistant.api.ingest import _safe_title
from research_assistant.vectorstore.client import similarity_search


# ── 1. Deterministic Document Hash & IDs ──────────────────────────────────────


def test_deterministic_document_hash() -> None:
    text1 = "Acme Corporation was founded in 2010."
    text2 = "Acme Corporation was founded in 2010."
    hash1 = compute_document_hash(text1)
    hash2 = compute_document_hash(text2)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex string length


def test_deterministic_document_id() -> None:
    doc_hash = compute_document_hash("Sample document text")
    doc_id1 = generate_document_id(doc_hash)
    doc_id2 = generate_document_id(doc_hash)
    assert doc_id1 == doc_id2


def test_deterministic_chunk_ids() -> None:
    doc_hash = compute_document_hash("Sample document text for chunking")
    chunk_id_0_a = generate_chunk_id(doc_hash, 0)
    chunk_id_0_b = generate_chunk_id(doc_hash, 0)
    chunk_id_1 = generate_chunk_id(doc_hash, 1)

    assert chunk_id_0_a == chunk_id_0_b
    assert chunk_id_0_a != chunk_id_1


# ── 2. Title Extraction Fix ───────────────────────────────────────────────────


def test_safe_title_preserves_numbers_in_filename() -> None:
    filename = "Acme Corporation was founded in 2010.txt"
    title = _safe_title(filename)
    assert title == "Acme Corporation was founded in 2010"


def test_safe_title_extracts_markdown_heading_for_generic_filename() -> None:
    filename = "upload.txt"
    text = "# Annual Financial Report 2025\n\nCompany revenue grew 20%."
    title = _safe_title(filename, text)
    assert title == "Annual Financial Report 2025"


# ── 3. Idempotent Chunking & Ingestion ────────────────────────────────────────


def test_chunk_text_produces_deterministic_metadata() -> None:
    text = "Line 1 text. Line 2 text."
    chunks1 = chunk_text(text=text, source="doc.txt")
    chunks2 = chunk_text(text=text, source="doc.txt")

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.document_id == c2.document_id
        assert c1.document_hash == c2.document_hash
        assert c1.chunk_id == c2.chunk_id
        assert c1.chunk_index == c2.chunk_index
        assert c1.text == c2.text


def test_upload_same_document_twice_returns_same_document_id(client: TestClient, monkeypatch) -> None:
    import research_assistant.api.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "validate_upload", lambda data, name: "text/plain")
    monkeypatch.setattr(ingest_mod, "extract_text", lambda data, mime: "Identical document content for testing idempotency.")
    monkeypatch.setattr(ingest_mod, "clean_text", lambda t: t)
    monkeypatch.setattr(ingest_mod, "get_embeddings", lambda texts: [[0.1] * 384 for _ in texts])
    monkeypatch.setattr(ingest_mod, "ensure_collection", lambda: None)
    monkeypatch.setattr(ingest_mod, "upsert_chunks", lambda chunks, vectors: len(chunks))

    file_bytes = b"Identical document content for testing idempotency."
    resp1 = client.post("/ingest/document", files={"file": ("doc.txt", file_bytes, "text/plain")})
    resp2 = client.post("/ingest/document", files={"file": ("doc.txt", file_bytes, "text/plain")})
    resp3 = client.post("/ingest/document", files={"file": ("doc.txt", file_bytes, "text/plain")})

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp3.status_code == 200

    body1, body2, body3 = resp1.json(), resp2.json(), resp3.json()
    assert body1["document_id"] == body2["document_id"] == body3["document_id"]
    assert body1["chunk_count"] == body2["chunk_count"] == body3["chunk_count"]


# ── 4. Retrieval Deduplication & Ranking ──────────────────────────────────────


def test_similarity_search_deduplicates_duplicate_points_preserving_rank(monkeypatch) -> None:
    import research_assistant.vectorstore.client as client_mod

    # Simulate Qdrant returning duplicate points (e.g. from legacy uploads)
    p1 = ScoredPoint(
        id="chunk-uuid-1",
        version=1,
        score=0.95,
        payload={"document_hash": "hash-A", "chunk_index": 0, "text": "First chunk content"},
    )
    p2 = ScoredPoint(
        id="chunk-uuid-2",
        version=1,
        score=0.95,
        payload={"document_hash": "hash-A", "chunk_index": 0, "text": "First chunk content"},
    )
    p3 = ScoredPoint(
        id="chunk-uuid-3",
        version=1,
        score=0.80,
        payload={"document_hash": "hash-A", "chunk_index": 1, "text": "Second chunk content"},
    )

    fake_client = MagicMock()
    fake_client.query_points.return_value = QueryResponse(points=[p1, p2, p3])
    monkeypatch.setattr(client_mod, "_get_client", lambda: fake_client)

    results = similarity_search(query_vector=[0.1] * 384, top_k=5, score_threshold=0.5)

    assert len(results) == 2
    assert results[0].id == "chunk-uuid-1"
    assert results[0].score == 0.95
    assert results[1].id == "chunk-uuid-3"
    assert results[1].score == 0.80


def test_deduplication_does_not_merge_different_documents_with_similar_text(monkeypatch) -> None:
    import research_assistant.vectorstore.client as client_mod

    p1 = ScoredPoint(
        id="chunk-uuid-1",
        version=1,
        score=0.90,
        payload={"document_hash": "hash-A", "chunk_index": 0, "text": "Table of Contents"},
    )
    p2 = ScoredPoint(
        id="chunk-uuid-2",
        version=1,
        score=0.88,
        payload={"document_hash": "hash-B", "chunk_index": 0, "text": "Table of Contents"},
    )

    fake_client = MagicMock()
    fake_client.query_points.return_value = QueryResponse(points=[p1, p2])
    monkeypatch.setattr(client_mod, "_get_client", lambda: fake_client)

    results = similarity_search(query_vector=[0.1] * 384, top_k=5, score_threshold=0.5)

    assert len(results) == 2
    assert results[0].payload["document_hash"] == "hash-A"
    assert results[1].payload["document_hash"] == "hash-B"
