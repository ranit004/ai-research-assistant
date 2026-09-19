"""Embedding generation using a local Sentence Transformers model.

The model (sentence-transformers/all-MiniLM-L6-v2) is downloaded on first use
and cached by the huggingface-hub cache.  No API key is required.

The public interface — ``get_embeddings(texts) -> list[list[float]]`` — is
identical to the previous OpenAI-backed version so callers require no changes.

Security:
- No API key is ever required or logged.
- The model is loaded lazily so tests that monkeypatch ``get_embeddings`` at
  the call-site never trigger a model download.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from research_assistant.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the SentenceTransformer model (called at most once per process)."""
    from sentence_transformers import SentenceTransformer  # lazy import

    model_name = settings.embedding_model
    logger.info("Loading local embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text using a local Sentence Transformers model.

    The model is loaded on first call and reused for subsequent calls.
    No external API key is required.

    Raises:
        RuntimeError: If the model cannot be loaded (e.g. package missing).
        Exception: Re-raises any other error so the caller can decide how to handle it.
    """
    try:
        model = _load_model()
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.debug("Embedding %d text(s) with model=%s", len(texts), settings.embedding_model)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [vec.tolist() for vec in embeddings]
