"""Uploaded PDFs and their sheets.

Upload limits come first: page count, not file size, is what sets the cost of a call. A PDF past
the page limit is refused whole, with no call and no charge, so the reader knows it was not read.
Documents live in memory only for the set being generated. Their local text is used only to look
for the quotes of each sheet, and to run a quote on to the end of its sentence for display.

What the page shows is not what the model reads: the model cites passages as E1, E2..., and the
page calls them P1, P2... and names each document by a short title instead of "Document N".
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
MAX_COMPLETION_WORDS = 40  # past this, a passage ends in "…" instead of its sentence's end
SHORT_TITLE_CHARS = 36
# A sentence ends at . ! or ? (and a closing quote or bracket) before a capital, a digit or the end.
SENTENCE_END = re.compile(r"[.!?][\"”’)\]]*(?=\s+[A-Z0-9(\[“\"]|\s*$)")
ABBREVIATIONS = ("e.g", "i.e", "et al", "al", "fig", "figs", "eq", "eqs", "vs", "etc", "cf", "sec")
DOCUMENT_REFERENCE = re.compile(r"\b[Dd]oc(?:ument)?s?\.?\s*#?(\d+)(?:\s*(?:and|&|,)\s*#?(\d+))?")


class UploadError(ValueError):
    """An upload that breaks a limit. The message is safe to show to the user."""


@dataclass(frozen=True)
class Document:
    number: int
    name: str
    data: bytes
    pages: int
    text: str  # normalized, to look for quotes
    cased: str = ""  # the same text with its case, to complete passages


@dataclass(frozen=True)
class Evidence:
    id: str
    quote: str
    found: bool  # the quote is in the text extracted locally from the PDF
    passage: str = ""  # the quote run on to the end of its sentence, or ending in "…"

    @property
    def label(self) -> str:
        return "P" + self.id.removeprefix("E")

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "quote": self.quote,
            "passage": self.passage or complete_passage(self.quote, ""),
            "found": self.found,
        }


@dataclass(frozen=True)
class Sheet:
    number: int
    name: str
    title: str
    summary: str
    methods: str
    limitations: str
    evidence: tuple[Evidence, ...]

    @property
    def short(self) -> str:
        return short_title(self.title)

    def public(self) -> dict[str, Any]:
        return {
            "document": self.number,
            "name": self.name,
            "title": self.title,
            "short": self.short,
            "summary": self.summary,
            "methods": self.methods,
            "limitations": self.limitations,
            "evidence": [e.public() for e in self.evidence],
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


def tidy(text: str) -> str:
    """Joins hyphenated line ends and collapses spacing, keeping case."""
    return re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", text)).strip()


def normalize(text: str) -> str:
    """For comparing a quote with the text of a PDF: joins hyphenated line ends and ignores case
    and spacing, because extraction rarely keeps either as printed."""
    return tidy(text).lower()


def _ends_sentence(text: str) -> bool:
    return text.rstrip('"”’)]').endswith((".", "!", "?"))


def complete_passage(quote: str, cased: str) -> str:
    """The quote run on to the end of its sentence in the document's text, if that is near; else the
    quote ending in "…" when it stops mid-sentence. Display only: the model reads the quote."""
    q = tidy(quote)
    if _ends_sentence(q):
        return q
    lowered = cased.lower()
    at = lowered.find(q.lower()) if len(lowered) == len(cased) else -1
    if at >= 0:
        end = at + len(q)
        rest = cased[end : end + 600]
        for m in SENTENCE_END.finditer(rest):
            before = rest[: m.start()].rsplit(" ", 1)[-1].lower().rstrip(".")
            if before in ABBREVIATIONS or (len(before) == 1 and before.isalpha()):
                continue  # "et al.", "Fig.", an initial: not a sentence end
            if len(rest[: m.end()].split()) <= MAX_COMPLETION_WORDS:
                return cased[at : end + m.end()]
            break
    return q.rstrip(" ,;:-") + "…"


def short_title(title: str) -> str:
    """A title for a source line: the part before a colon, or the title cut at a word."""
    t = " ".join(title.split())
    head = t.split(":", 1)[0].strip()
    if ":" in t and 0 < len(head) <= SHORT_TITLE_CHARS:
        return head
    if len(t) <= SHORT_TITLE_CHARS:
        return t
    return t[: SHORT_TITLE_CHARS + 1].rsplit(" ", 1)[0].rstrip(",;:-") + "…"


def name_documents(text: str, shorts: dict[int, str]) -> str:
    """ "Document 2", "Documents 1 and 2"... replaced by the short titles, for display."""

    def replace(m: re.Match) -> str:
        names = [shorts.get(int(n)) for n in m.groups() if n]
        return m.group(0) if None in names else " and ".join(names)

    return DOCUMENT_REFERENCE.sub(replace, text)


IDEA_FIELDS = ("problem_gap", "core_claim", "method", "inspiration")


def display_idea(idea: dict[str, Any], sheets: list[Sheet]) -> dict[str, str]:
    """The fields of an idea as the page shows them, with each document named by its short title.
    The idea itself, which later calls read, is left as the model wrote it."""
    shorts = {s.number: s.short for s in sheets}
    return {f: name_documents(idea[f], shorts) for f in IDEA_FIELDS if f in idea}


def source_item(sheet: Sheet, evidence: Evidence) -> dict[str, Any]:
    """A cited passage as the page shows it: "<short title> · P6", with the completed passage."""
    return {
        "document": sheet.number,
        "name": sheet.name,
        "short": sheet.short,
        "evidence": evidence.id,
        "label": evidence.label,
        "quote": evidence.quote,
        "passage": evidence.public()["passage"],
        "found": evidence.found,
    }


def read_pdf(data: bytes, shown: str, max_pages: int) -> tuple[int, str]:
    """Pages and the tidied text, with its case."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadError(f"{shown}: password-protected PDFs are not supported.")
        pages = len(reader.pages)
        if pages == 0:
            raise UploadError(f"{shown}: the PDF has no pages.")
        if pages > max_pages:
            raise UploadError(f"{shown}: {pages} pages; the limit is {max_pages} per document.")
        return pages, tidy("\n".join(page.extract_text() or "" for page in reader.pages))
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
        pages, cased = read_pdf(data, shown, limits["max_pages_per_document"])
        documents.append(Document(number, shown, data, pages, cased.lower(), cased))
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
            Evidence(
                f"E{i}",
                q,
                normalize(q) in document.text,
                complete_passage(q, document.cased),
            )
            for i, q in enumerate(quotes, 1)
        ),
    )
