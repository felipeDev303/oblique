"""Structural validity of a generated idea set (ADR-006, section 1).

An idea is invalid if the response does not parse as JSON, the set does not
contain exactly K cards, a required field is missing, or a field is empty or
whitespace only. Invalid ideas get q = 0, are not judged and count in the
invalid rate. Label conformity (section 2) never decides validity.

Accepted shapes: a JSON array of K idea objects, or an object whose "ideas"
key holds that array. The text must be JSON as-is (surrounding whitespace aside).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from oblique.generate.schema import IDEA_FIELDS, IdeaCard


class InvalidReason(StrEnum):
    json_error = "json_error"  # response does not parse as JSON (the only retryable case)
    bad_shape = "bad_shape"  # JSON, but not an array of ideas
    wrong_count = "wrong_count"  # not exactly K ideas
    not_object = "not_object"  # an element of the array is not an object
    missing_field = "missing_field"
    not_text = "not_text"  # field present but not a string
    empty_field = "empty_field"  # field empty or whitespace only


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


def _all_invalid(k: int, reason: InvalidReason, detail: str) -> ParsedSet:
    return ParsedSet(k, tuple(Slot(i, None, reason, detail) for i in range(k)))


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


def parse_set(text: str | None, k: int) -> ParsedSet:
    """Parse a model response into K slots, each a valid idea or an invalid reason.

    Set-level failures (unparseable JSON, wrong shape, wrong count) invalidate
    all K slots. Field-level failures invalidate only that idea.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    try:
        data = json.loads(text or "")
    except json.JSONDecodeError as e:
        return _all_invalid(k, InvalidReason.json_error, str(e))

    if isinstance(data, dict) and "ideas" in data:
        data = data["ideas"]
    if not isinstance(data, list):
        return _all_invalid(k, InvalidReason.bad_shape, type(data).__name__)
    if len(data) != k:
        return _all_invalid(k, InvalidReason.wrong_count, f"expected {k}, got {len(data)}")
    return ParsedSet(k, tuple(_check_idea(i, obj) for i, obj in enumerate(data)))
