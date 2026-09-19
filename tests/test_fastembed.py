"""Focused tests for FastEmbed integration, embedding generation, dimension verification,
Qdrant collection initialization, and retrieval compatibility.
"""

from unittest.mock import MagicMock, patch
import pytest

from research_assistant.config import settings
from research_assistant.vectorstore.embedder import (
    get_embedding_dimension,
    get_embeddings,
)
from research_assistant.vectorstore.client import ensure_collection, similarity_search


# ── 1. Embedding generation (Mocked unit test) ─────────────────────────────


def test_get_embeddings_generation_mocked() -> None:
    fake_model = MagicMock()
    # FastEmbed model.embed returns generator yielding numpy arrays
    fake_vec = MagicMock()
    fake_vec.tolist.return_value = [0.1] * 384
    fake_model.embed.return_value = [fake_vec]

    with patch("research_assistant.vectorstore.embedder._load_model", return_value=fake_model):
        result = get_embeddings(["Hello world"])
        assert len(result) == 1
        assert len(result[0]) == 384
        assert result[0][0] == 0.1


# ── 2. Embedding dimension ─────────────────────────────────────────────────


def test_get_embedding_dimension() -> None:
    fake_model = MagicMock()
    fake_model.embedding_dimension = 384

    with patch("research_assistant.vectorstore.embedder._load_model", return_value=fake_model):
        dim = get_embedding_dimension()
        assert dim == 384


# ── 3. Multiple document embeddings ────────────────────────────────────────


def test_get_embeddings_multiple_documents() -> None:
    fake_model = MagicMock()
    fake_vec1 = MagicMock()
    fake_vec1.tolist.return_value = [0.1] * 384
    fake_vec2 = MagicMock()
    fake_vec2.tolist.return_value = [0.2] * 384
    fake_model.embed.return_value = [fake_vec1, fake_vec2]

    with patch("research_assistant.vectorstore.embedder._load_model", return_value=fake_model):
        result = get_embeddings(["Doc 1 text", "Doc 2 text"])
        assert len(result) == 2
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2


# ── 4. Retrieval compatibility ─────────────────────────────────────────────


def test_similarity_search_compatibility() -> None:
    fake_client = MagicMock()
    fake_point = MagicMock()
    fake_point.id = "c1"
    fake_point.score = 0.95
    fake_point.payload = {"text": "retrieved text", "source": "test.txt"}
    fake_response = MagicMock()
    fake_response.points = [fake_point]
    fake_client.query_points.return_value = fake_response

    with patch("research_assistant.vectorstore.client._get_client", return_value=fake_client):
        query_vec = [0.1] * 384
        hits = similarity_search(query_vec, top_k=5, score_threshold=0.3)
        assert len(hits) == 1
        assert hits[0].payload["text"] == "retrieved text"
        fake_client.query_points.assert_called_once_with(
            collection_name=settings.qdrant_collection,
            query=query_vec,
            limit=15,
            score_threshold=0.3,
            with_payload=True,
        )


# ── 5. Qdrant collection initialization & dimension mismatch recreation ────


def test_ensure_collection_recreates_on_dimension_mismatch() -> None:
    fake_client = MagicMock()
    # Mock existing collections list
    coll_info = MagicMock()
    coll_info.name = settings.qdrant_collection
    fake_client.get_collections.return_value.collections = [coll_info]

    # Mock existing collection details with OLD incompatible vector size (e.g. 768)
    old_coll_detail = MagicMock()
    old_coll_detail.config.params.vectors.size = 768
    fake_client.get_collection.return_value = old_coll_detail

    with patch("research_assistant.vectorstore.client._get_client", return_value=fake_client):
        ensure_collection()
        # Verify existing incompatible collection was deleted and recreated with 384
        fake_client.delete_collection.assert_called_once_with(collection_name=settings.qdrant_collection)
        fake_client.create_collection.assert_called_once()


# ── 6. Unsupported query behavior ──────────────────────────────────────────


def test_empty_embeddings_handles_empty_input() -> None:
    assert get_embeddings([]) == []


# ── 7. Real local smoke test using actual FastEmbed model ─────────────────


def test_fastembed_real_smoke_test() -> None:
    """Smoke test that executes the actual FastEmbed model without mocks."""
    embeddings = get_embeddings(["FastEmbed smoke test query"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384
    assert isinstance(embeddings[0][0], float)
