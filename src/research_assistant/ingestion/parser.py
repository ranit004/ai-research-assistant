"""File-type detection and text extraction.

Security rules applied here:
- MIME type is detected from magic bytes, not from the user-supplied filename.
- The filename is never used to construct a filesystem path.
- Stack traces are never propagated to callers; only a ValueError is raised.
"""

import io
import logging
import re
import unicodedata

import magic
from pypdf import PdfReader

from research_assistant.config import settings

logger = logging.getLogger(__name__)


def detect_mime(data: bytes) -> str:
    """Return MIME type detected from file magic bytes."""
    return magic.from_buffer(data, mime=True)


def validate_upload(data: bytes, original_filename: str) -> str:
    """Check size and MIME type.  Return the detected MIME type on success.

    Raises:
        ValueError: with a user-safe message when validation fails.
    """
    if len(data) > settings.max_upload_bytes:
        raise ValueError(
            f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit."
        )
    mime = detect_mime(data)
    if mime not in settings.allowed_mime_types:
        raise ValueError(
            f"Unsupported file type '{mime}'. "
            f"Allowed: {', '.join(settings.allowed_mime_types)}."
        )
    logger.debug("Upload validated: mime=%s bytes=%d", mime, len(data))
    return mime


def extract_text(data: bytes, mime: str) -> str:
    """Extract plain text from *data* according to its *mime* type.

    Returns the extracted string.  Never raises; logs a warning and returns an
    empty string if extraction fails.
    """
    try:
        if mime == "application/pdf":
            return _extract_pdf(data)
        # text/plain and text/markdown
        return data.decode("utf-8", errors="replace")
    except Exception:
        logger.warning("Text extraction failed for mime=%s", mime, exc_info=True)
        return ""


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    """Normalise whitespace and remove non-printable and dangerous Unicode characters."""
    # NFC normalisation so accented chars are consistent
    text = unicodedata.normalize("NFC", raw)
    # Strip Unicode bidi overrides, invisible markers, and private-use codepoints
    # that could be used to smuggle hidden instructions through text displays.
    text = re.sub(
        r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u206a-\u206f\ufff0-\uffff]",
        "",
        text,
    )
    # Replace form feeds, vertical tabs, etc. with a newline
    text = re.sub(r"[\x0b\x0c\r]", "\n", text)
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()
