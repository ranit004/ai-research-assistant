"""POST /documents — document upload endpoint alias."""

from fastapi import APIRouter, UploadFile, status

from research_assistant.api.ingest import IngestResponse, ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile) -> IngestResponse:
    """Accept a document upload, parse, chunk, embed, and index it into Qdrant."""
    return await ingest_document(file)
