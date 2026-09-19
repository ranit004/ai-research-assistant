"""Fixed-size character-level chunking with overlap and deterministic identity generation.

Character-level splitting is used rather than token-level because it has
zero additional dependencies and behaves predictably across all languages.
Deterministic UUID5 generation guarantees idempotent document ingestion.
"""

import hashlib
import uuid

from research_assistant.config import settings
from research_assistant.ingestion.models import DocumentChunk


def compute_document_hash(text: str) -> str:
    """Return a SHA-256 hex digest of the normalized document text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_document_id(document_hash: str) -> str:
    """Return a deterministic UUID5 string for a document based on its SHA-256 hash."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{document_hash}"))


def generate_chunk_id(document_hash: str, chunk_index: int) -> str:
    """Return a deterministic UUID5 string for a chunk based on doc hash and chunk index."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{document_hash}:{chunk_index}"))


def chunk_text(
    text: str,
    document_id: str | None = None,
    title: str = "untitled",
    source: str = "upload",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    document_hash: str | None = None,
) -> list[DocumentChunk]:
    """Split *text* into overlapping chunks and attach deterministic metadata.

    Args:
        text: Clean document text.
        document_id: Optional UUID string for the parent document (derived if omitted).
        document_hash: Optional SHA-256 hash of the document text.
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

    doc_hash = document_hash or compute_document_hash(text)
    doc_id = document_id or generate_document_id(doc_hash)

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
                    document_id=doc_id,
                    document_hash=doc_hash,
                    chunk_id=generate_chunk_id(doc_hash, index),
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
