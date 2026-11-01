"""Checks on uploaded PDFs, before anything reaches the model.

The limits protect the credits: the provider reads the text of up to 100 pages per PDF, so page
count, not file size, is what sets the cost of a call. Documents live in memory only for the set
being generated. Their local text is used only to look for the quotes the model cites.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class UploadError(ValueError):
    """An upload that breaks a limit. The message is safe to show to the user."""


@dataclass(frozen=True)
class Document:
    number: int
    name: str
    data: bytes
    pages: int
    text: str


def clean_name(name: str | None, number: int) -> str:
    """The name to display: no path, no control characters, bounded."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:120] or f"document-{number}.pdf"


def normalize(text: str) -> str:
    """For comparing a quote with the text of a PDF: joins hyphenated line ends and ignores case
    and spacing, because extraction rarely keeps either as printed."""
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", text)).strip().lower()


def read_pdf(data: bytes, shown: str, max_pages: int) -> tuple[int, str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError(f"{shown}: password-protected PDFs are not supported.")
        pages = len(reader.pages)
        if pages == 0:
            raise UploadError(f"{shown}: the PDF has no pages.")
        if pages > max_pages:
            raise UploadError(f"{shown}: {pages} pages; the limit is {max_pages} per document.")
        return pages, normalize("\n".join(page.extract_text() or "" for page in reader.pages))
    except UploadError:
        raise
    except (PdfReadError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        raise UploadError(f"{shown}: the PDF could not be read.") from e


def load_documents(files: list[tuple[str | None, bytes]], limits: dict[str, Any]) -> list[Document]:
    """Validates a whole upload, or raises `UploadError` on the first broken limit."""
    low, high = limits["min_documents"], limits["max_documents"]
    if not low <= len(files) <= high:
        raise UploadError(f"Upload between {low} and {high} PDFs; got {len(files)}.")
    max_bytes = limits["max_file_mb"] * 1_000_000
    documents, seen = [], set()
    for number, (name, data) in enumerate(files, 1):
        shown = clean_name(name, number)
        if len(data) > max_bytes:
            raise UploadError(f"{shown}: larger than {limits['max_file_mb']} MB.")
        if not data.startswith(b"%PDF-"):
            raise UploadError(f"{shown}: not a PDF.")
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            raise UploadError(f"{shown}: the same file was uploaded twice.")
        seen.add(digest)
        pages, text = read_pdf(data, shown, limits["max_pages_per_document"])
        documents.append(Document(number, shown, data, pages, text))
    return documents
