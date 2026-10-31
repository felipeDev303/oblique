"""Structural validity of a generated idea set (ADR-006 §1 and its 2026-10-30 amendments).

An idea is invalid if the response does not parse as JSON, the set does not
contain exactly K cards, a required field is missing, not text, or empty /
whitespace only. Invalid ideas get q = 0, are not judged and count in the
invalid rate. Label conformity (§2) never decides validity.

Accepted shapes: a JSON array of K idea objects, or an object whose "ideas"
key holds that array; which one arrived is recorded as `shape`. The text must
be JSON as-is (surrounding whitespace aside), or exactly one ```json / ```
code block, in which case the wrapper is removed and `fenced` is recorded.
Any other text around the JSON is a `json_error`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from oblique.generate.schema import IDEA_FIELDS, IdeaCard

# One whole-response code block: opening ```json or ``` on its own line, closing ``` at the end.
_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(.*?)\r?\n?```\Z", re.DOTALL)

Shape = Literal["list", "ideas_object"]


class InvalidReason(StrEnum):
    json_error = "json_error"  # response does not parse as JSON (the only retryable case)
    bad_shape = "bad_shape"  # JSON, but not an array of ideas
    wrong_count = "wrong_count"  # not exactly K ideas
    not_object = "not_object"  # an element of the array is not an object
    missing_field = "missing_field"
    not_text = "not_text"  # field present but not a string
    empty_field = "empty_field"  # field empty or whitespace only
    content_filtered = "content_filtered"  # the provider's filter refused the request twice


@dataclass(frozen=True)
class Slot:
    index: int
    idea: IdeaCard | None
    reason: InvalidReason | None = None
    detail: str | None = None

    @property
    def valid(self) -> bool:
        return self.idea is not None


@dataclass(frozen=True)
class ParsedSet:
    k: int
    slots: tuple[Slot, ...]
    fenced: bool = False
    shape: Shape | None = None  # None when the response is not a recognised shape

    @property
    def valid(self) -> list[IdeaCard]:
        return [s.idea for s in self.slots if s.idea is not None]

    @property
    def invalid(self) -> list[Slot]:
        return [s for s in self.slots if s.idea is None]

    @property
    def retryable(self) -> bool:
        """True only when the response did not parse as JSON."""
        return any(s.reason is InvalidReason.json_error for s in self.slots)


def unfence(text: str) -> tuple[str, bool]:
    """Strip a single whole-response ```json / ``` block. Returns (text, fenced)."""
    stripped = text.strip()
    match = _FENCE.match(stripped)
    if match is None or "```" in match.group(1):
        return stripped, False
    return match.group(1), True


def api_seed(sample_idx: int, attempt: int) -> int:
    """Seed sent to the API (only if it honours `seed`): 1000 * attempt + sample_idx."""
    if not 0 <= sample_idx < 1000 or attempt < 0:
        raise ValueError("need 0 <= sample_idx < 1000 and attempt >= 0")
    return 1000 * attempt + sample_idx


def _check_idea(index: int, obj: Any) -> Slot:
    if not isinstance(obj, dict):
        return Slot(index, None, InvalidReason.not_object, type(obj).__name__)
    for field in IDEA_FIELDS:
        if field not in obj:
            return Slot(index, None, InvalidReason.missing_field, field)
        if not isinstance(obj[field], str):
            return Slot(index, None, InvalidReason.not_text, field)
        if not obj[field].strip():
            return Slot(index, None, InvalidReason.empty_field, field)
    return Slot(index, IdeaCard.model_validate(obj))


def filtered_set(k: int) -> ParsedSet:
    """A generation the content filter refused: K invalid slots (ADR-006, amendment 6)."""
    slots = tuple(Slot(i, None, InvalidReason.content_filtered) for i in range(k))
    return ParsedSet(k, slots)


def parse_set(text: str | None, k: int) -> ParsedSet:
    """Parse a model response into K slots, each a valid idea or an invalid reason.

    Set-level failures (unparseable JSON, wrong shape, wrong count) invalidate
    all K slots. Field- or element-level failures invalidate only that idea.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    body, fenced = unfence(text or "")

    def all_invalid(reason: InvalidReason, detail: str, shape: Shape | None = None) -> ParsedSet:
        slots = tuple(Slot(i, None, reason, detail) for i in range(k))
        return ParsedSet(k, slots, fenced=fenced, shape=shape)

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return all_invalid(InvalidReason.json_error, str(e))

    shape: Shape | None
    if isinstance(data, list):
        shape = "list"
    elif isinstance(data, dict) and isinstance(data.get("ideas"), list):
        shape, data = "ideas_object", data["ideas"]
    else:
        return all_invalid(InvalidReason.bad_shape, type(data).__name__)
    if len(data) != k:
        return all_invalid(InvalidReason.wrong_count, f"expected {k}, got {len(data)}", shape)
    slots = tuple(_check_idea(i, obj) for i, obj in enumerate(data))
    return ParsedSet(k, slots, fenced=fenced, shape=shape)
