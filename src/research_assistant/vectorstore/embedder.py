"""Embedding generation.

The embedder wraps langchain-openai's `OpenAIEmbeddings`.  When no OpenAI key
is configured the module raises `RuntimeError` at call time, not at import
time, so the rest of the application (and tests that mock the embedder) can
still import without a live key.
"""

import logging

from research_assistant.config import settings

logger = logging.getLogger(__name__)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text.

    Raises:
        RuntimeError: If no `OPENAI_API_KEY` is configured.
        Exception: Re-raises any upstream API error so the caller can decide
            how to handle it.
    """
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Set it in your .env file or environment."
        )

    from langchain_openai import OpenAIEmbeddings  # lazy import

    embedder = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key.get_secret_value(),
    )
    logger.debug("Embedding %d text(s) with model=%s", len(texts), settings.embedding_model)
    return embedder.embed_documents(texts)
