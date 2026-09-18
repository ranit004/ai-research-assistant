"""POST /ingest/document — upload a document, parse, chunk, embed, and store.

Security controls:
- Size validated before any parsing.
- MIME type detected from magic bytes; the user filename is sanitised and
  never used as a filesystem path.
- Secrets are never logged.
- Internal errors return 500 with a generic message; no stack traces leak.
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from research_assistant.ingestion.chunker import chunk_text
from research_assistant.ingestion.parser import clean_text, extract_text, validate_upload
from research_assistant.vectorstore.client import ensure_collection, upsert_chunks
from research_assistant.vectorstore.embedder import get_embeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

_MAX_FILENAME_LENGTH = 255
_MAX_CHUNKS_PER_DOCUMENT = 500   # prevents a 10 MB doc producing ~19k embedding calls


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    title: str


def _safe_title(filename: str) -> str:
    """Derive a display title from the uploaded filename.

    The result is used only as metadata; it is never used as a filesystem path.
    """
    # Keep only the basename (no directory separators)
    name = filename.replace("\\", "/").split("/")[-1]
    # Strip the extension for a clean title
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name[:_MAX_FILENAME_LENGTH] or "untitled"


@router.post("/document", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(file: UploadFile) -> IngestResponse:
    """Accept a document upload, parse and chunk it, embed each chunk, and
    store the results in Qdrant.

    Allowed types: PDF, plain text, Markdown (max 10 MB by default).
    """
    raw = await file.read()
    original_name = file.filename or "upload"

    try:
        mime = validate_upload(raw, original_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    document_id = str(uuid.uuid4())
    title = _safe_title(original_name)

    text = extract_text(raw, mime)
    text = clean_text(text)

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No extractable text found in the uploaded document.",
        )

    chunks = chunk_text(
        text=text,
        document_id=document_id,
        title=title,
        source=original_name[:_MAX_FILENAME_LENGTH],
    )

    if len(chunks) > _MAX_CHUNKS_PER_DOCUMENT:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Document produces {len(chunks)} chunks, which exceeds the "
                f"{_MAX_CHUNKS_PER_DOCUMENT}-chunk limit. "
                "Please split the document into smaller files."
            ),
        )

    try:
        vectors = get_embeddings([c.text for c in chunks])
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Embedding generation failed for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service error. Please try again later.",
        )

    try:
        ensure_collection()
        upsert_chunks(chunks, vectors)
    except Exception:
        logger.exception("Qdrant upsert failed for document_id=%s", document_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store error. Please try again later.",
        )

    logger.info(
        "Ingested document_id=%s title=%r chunks=%d",
        document_id,
        title,
        len(chunks),
    )
    return IngestResponse(document_id=document_id, chunk_count=len(chunks), title=title)
