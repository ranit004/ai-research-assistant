"""Tests for the ingestion pipeline: parser, chunker, and models.

These tests are pure unit tests — they do not need a running Qdrant instance
or an OpenAI key.
"""

import pytest

from research_assistant.ingestion.chunker import chunk_text, _detect_section
from research_assistant.ingestion.models import DocumentChunk
from research_assistant.ingestion.parser import clean_text, validate_upload


# ── clean_text ─────────────────────────────────────────────────────────────────


def test_clean_text_strips_trailing_whitespace() -> None:
    result = clean_text("hello   \nworld   ")
    assert result == "hello\nworld"


def test_clean_text_collapses_blank_lines() -> None:
    result = clean_text("a\n\n\n\nb")
    assert result == "a\n\nb"


def test_clean_text_handles_windows_line_endings() -> None:
    result = clean_text("line1\r\nline2")
    assert "line1" in result
    assert "line2" in result


# ── validate_upload ────────────────────────────────────────────────────────────


def test_validate_upload_rejects_oversized_file(monkeypatch) -> None:
    import research_assistant.ingestion.parser as parser_mod
    monkeypatch.setattr(parser_mod.settings, "max_upload_bytes", 10)
    with pytest.raises(ValueError, match="exceeds"):
        validate_upload(b"x" * 11, "test.txt")


def test_validate_upload_rejects_disallowed_mime(monkeypatch, tmp_path) -> None:
    # python-magic may not be available in all CI environments; mock it.
    import research_assistant.ingestion.parser as parser_mod
    monkeypatch.setattr(parser_mod, "detect_mime", lambda data: "image/png")
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_upload(b"fake-png-data", "test.png")


def test_validate_upload_accepts_plain_text(monkeypatch) -> None:
    import research_assistant.ingestion.parser as parser_mod
    monkeypatch.setattr(parser_mod, "detect_mime", lambda data: "text/plain")
    mime = validate_upload(b"hello world", "test.txt")
    assert mime == "text/plain"


# ── chunk_text ─────────────────────────────────────────────────────────────────


def test_chunk_text_produces_correct_metadata() -> None:
    text = "a" * 600
    chunks = chunk_text(text, "doc-1", "Test Title", "test.txt", chunk_size=200, chunk_overlap=0)
    assert len(chunks) == 3
    for i, chunk in enumerate(chunks):
        assert chunk.document_id == "doc-1"
        assert chunk.title == "Test Title"
        assert chunk.source == "test.txt"
        assert chunk.chunk_index == i
        assert chunk.chunk_id  # non-empty UUID


def test_chunk_text_with_overlap() -> None:
    text = "abcdefghij"
    chunks = chunk_text(text, "d", "T", "f.txt", chunk_size=6, chunk_overlap=2)
    # step = 4; starts at 0, 4, 8 -> 3 chunks (last is "ij", non-empty so kept)
    assert len(chunks) == 3
    assert chunks[0].text == "abcdef"
    assert chunks[1].text == "efghij"
    assert chunks[2].text == "ij"


def test_chunk_text_empty_string_returns_no_chunks() -> None:
    chunks = chunk_text("", "d", "T", "f.txt")
    assert chunks == []


def test_chunk_text_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", "d", "T", "f.txt", chunk_size=0)


def test_chunk_text_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", "d", "T", "f.txt", chunk_size=10, chunk_overlap=10)


# ── _detect_section ────────────────────────────────────────────────────────────


def test_detect_section_finds_heading() -> None:
    text = "# Introduction\nSome content here."
    # position is after the heading
    section = _detect_section(text, 20)
    assert section == "Introduction"


def test_detect_section_no_heading_returns_empty() -> None:
    section = _detect_section("plain text with no headings", 5)
    assert section == ""


# ── DocumentChunk model ────────────────────────────────────────────────────────


def test_document_chunk_serialises_correctly() -> None:
    chunk = DocumentChunk(
        document_id="d1",
        chunk_id="c1",
        chunk_index=0,
        title="My Doc",
        section="Intro",
        source="mydoc.txt",
        text="Hello world",
    )
    data = chunk.model_dump()
    assert data["document_id"] == "d1"
    assert data["section"] == "Intro"
    assert data["text"] == "Hello world"
