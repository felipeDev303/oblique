"""Uploaded PDFs and their sheets.

Upload limits come first: page count, not file size, is what sets the cost of a call. A PDF past
the page limit is refused whole, with no call and no charge, so the reader knows it was not read.
Documents live in memory only for the set being generated. Their local text is used only to look
for the quotes of each sheet.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, field_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from oblique.prompts import JudgeOutputError
from oblique.prompts.base import parse_json_object

MAX_EVIDENCE = 10


class UploadError(ValueError):
    """An upload that breaks a limit. The message is safe to show to the user."""


@dataclass(frozen=True)
class Document:
    number: int
    name: str
    data: bytes
    pages: int
    text: str


@dataclass(frozen=True)
class Evidence:
    id: str
    quote: str
    found: bool  # the quote is in the text extracted locally from the PDF


@dataclass(frozen=True)
class Sheet:
    number: int
    name: str
    title: str
    summary: str
    methods: str
    limitations: str
    evidence: tuple[Evidence, ...]

    def public(self) -> dict[str, Any]:
        return {
            "document": self.number,
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "methods": self.methods,
            "limitations": self.limitations,
            "evidence": [vars(e) for e in self.evidence],
        }


class SheetReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: StrictStr
    summary: StrictStr
    methods: StrictStr
    limitations: StrictStr
    evidence: list[StrictStr]

    @field_validator("title", "summary")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field is empty or whitespace only")
        return v


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


def parse_sheet(content: str | None, document: Document) -> Sheet | None:
    """The sheet of a reply, or None if it breaks the contract. A quote not found in the local text
    stays, marked: it may be real (scanned pages, odd extraction)."""
    try:
        reply = SheetReply.model_validate(parse_json_object(content))
    except (JudgeOutputError, ValidationError):
        return None
    quotes = [q for q in reply.evidence if q.strip()][:MAX_EVIDENCE]
    if not quotes:
        return None
    return Sheet(
        number=document.number,
        name=document.name,
        title=reply.title,
        summary=reply.summary,
        methods=reply.methods,
        limitations=reply.limitations,
        evidence=tuple(
            Evidence(f"E{i}", q, normalize(q) in document.text) for i, q in enumerate(quotes, 1)
        ),
    )
