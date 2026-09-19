"""Shared Pydantic models for document ingestion."""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """One chunk produced from an ingested document.

    Every field is required so retrieval results are always complete.
    """

    document_id: str = Field(description="Deterministic UUID assigned to the uploaded document.")
    document_hash: str = Field(default="", description="SHA-256 hash of the normalized document text.")
    chunk_id: str = Field(description="Deterministic UUID assigned to this individual chunk.")
    chunk_index: int = Field(description="Zero-based position of this chunk within the document.")
    title: str = Field(description="Title derived from the document or filename stub.")
    section: str = Field(default="", description="Section heading, when extractable.")
    source: str = Field(description="Original filename as supplied by the uploader (sanitised, not used as a path).")
    text: str = Field(description="The raw chunk text.")
