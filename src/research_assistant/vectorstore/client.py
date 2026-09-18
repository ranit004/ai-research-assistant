"""Qdrant collection management, upsert, and similarity search.

The client is created lazily (on first call) so the application starts even
when Qdrant is not yet reachable.  Integration tests that need a real client
should start a Qdrant instance (e.g. via Docker) and set QDRANT_URL.
"""

import logging
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from research_assistant.config import settings
from research_assistant.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    """Return (and cache) a Qdrant client.

    The API key is passed only if it is set; an unauthenticated local Qdrant
    instance works without it.
    """
    kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key.get_secret_value()
    logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
    return QdrantClient(**kwargs)


def ensure_collection() -> None:
    """Create the collection if it does not already exist."""
    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'.", settings.qdrant_collection)
    else:
        logger.debug("Qdrant collection '%s' already exists.", settings.qdrant_collection)


def upsert_chunks(chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
    """Store chunk payloads and their vectors in Qdrant.

    Args:
        chunks: DocumentChunk objects to store.
        vectors: One embedding vector per chunk (same order).

    Returns:
        Number of points upserted.

    Raises:
        ValueError: If the length of *chunks* and *vectors* differ.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length.")

    client = _get_client()
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload=chunk.model_dump(),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Upserted %d points into '%s'.", len(points), settings.qdrant_collection)
    return len(points)


def similarity_search(
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[ScoredPoint]:
    """Return the *top_k* most similar chunks.

    Args:
        query_vector: Embedding of the query string.
        top_k: Maximum number of results.
        score_threshold: Minimum cosine similarity score (0–1).

    Returns:
        List of :class:`ScoredPoint` objects, sorted by descending score.
    """
    if top_k < 1 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100.")

    client = _get_client()
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    logger.debug("Similarity search returned %d results.", len(results))
    return results
