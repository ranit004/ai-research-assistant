"""POST /retrieve — similarity search over stored document chunks."""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from research_assistant.vectorstore.client import similarity_search
from research_assistant.vectorstore.embedder import get_embeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Natural-language search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return.")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score.",
    )


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    title: str
    section: str
    source: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    results: list[ChunkResult]


@router.post("", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Return the most relevant document chunks for a query."""
    try:
        query_vector = get_embeddings([request.query])[0]
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Embedding failed during retrieval.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding service error.",
        )

    try:
        hits = similarity_search(
            query_vector=query_vector,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except Exception:
        logger.exception("Qdrant search failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vector store error.",
        )

    results = [
        ChunkResult(
            chunk_id=str(hit.id),
            score=hit.score,
            **{k: hit.payload[k] for k in ("document_id", "chunk_index", "title", "section", "source", "text")},
        )
        for hit in hits
        if hit.payload
    ]
    return RetrieveResponse(results=results)
