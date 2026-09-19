"""Fixed-size character-level chunking with overlap.

Character-level splitting is used rather than token-level because it has
zero additional dependencies and behaves predictably across all languages.
Token counts are intentionally not estimated here to avoid coupling to a
specific tokeniser.
"""

import uuid

from research_assistant.config import settings
from research_assistant.ingestion.models import DocumentChunk


def chunk_text(
    text: str,
    document_id: str,
    title: str,
    source: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Split *text* into overlapping chunks and attach metadata.

    Args:
        text: Clean document text.
        document_id: UUID string for the parent document.
        title: Human-readable title (derived from filename or document header).
        source: Original filename supplied by the uploader.
        chunk_size: Characters per chunk.  Defaults to `settings.chunk_size`.
        chunk_overlap: Characters of overlap.  Defaults to `settings.chunk_overlap`.

    Returns:
        Ordered list of :class:`DocumentChunk` objects.
    """
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    chunks: list[DocumentChunk] = []
    step = size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = start + size
        span = text[start:end].strip()
        if span:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_id=str(uuid.uuid4()),
                    chunk_index=index,
                    title=title,
                    section=_detect_section(text, start),
                    source=source,
                    text=span,
                )
            )
            index += 1
        start += step

    return chunks


def _detect_section(text: str, position: int) -> str:
    """Return the nearest Markdown heading before *position*, or empty string."""
    preceding = text[:position]
    lines = preceding.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""
