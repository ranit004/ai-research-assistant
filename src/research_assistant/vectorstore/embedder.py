"""Embedding generation using Qdrant FastEmbed (ONNX Runtime).

The model (default: BAAI/bge-small-en-v1.5) is loaded lazily and cached locally.
It produces 384-dimensional dense vectors with minimal memory footprint (<100MB RAM),
making it suitable for Render Free (512MB RAM limit).

Public interfaces:
- get_embeddings(texts) -> list[list[float]]
- get_embedding_dimension() -> int
"""

from __future__ import annotations

import logging
from functools import lru_cache

from research_assistant.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the FastEmbed TextEmbedding model (called at most once per process)."""
    try:
        from fastembed import TextEmbedding  # lazy import
    except ImportError as exc:
        raise RuntimeError(
            "fastembed is not installed. "
            "Install it with: pip install fastembed"
        ) from exc

    model_name = settings.embedding_model
    logger.info("Loading local FastEmbed model: %s", model_name)
    return TextEmbedding(model_name=model_name)


def get_embedding_dimension() -> int:
    """Return the vector dimension of the active embedding model."""
    try:
        model = _load_model()
        if hasattr(model, "embedding_dimension"):
            return model.embedding_dimension
        vecs = list(model.embed(["test"]))
        return len(vecs[0])
    except Exception:
        return settings.embedding_dim


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text using Qdrant FastEmbed.

    The model is loaded on first call and reused for subsequent calls.
    No external API key is required.

    Args:
        texts: List of strings to generate embeddings for.

    Returns:
        List of float vectors, one per input text.
    """
    if not texts:
        return []

    model = _load_model()
    logger.debug("Embedding %d text(s) with FastEmbed model=%s", len(texts), settings.embedding_model)
    embeddings = list(model.embed(texts))
    return [vec.tolist() for vec in embeddings]
