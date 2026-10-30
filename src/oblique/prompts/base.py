"""Versioned prompt templates and the helpers they share.

A prompt is an ID (`name-vN`), a fixed system message and a user template.
Its hash covers all three and goes into `prompt_hash`, so any edit to a
template must come with a new version. Every prompt puts what is shared across
calls first (system, then the source papers) so the provider's prompt cache
can reuse that prefix.

These prompts were written for this project from a specification (ADR-007).
They follow the structure of Wadhwa et al. (2026) and the taxonomy of
Chen et al. (2026); no text is taken from either paper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol

from oblique.generate.validate import unfence


class JudgeOutputError(ValueError):
    """A judge response that does not match its prompt's output contract."""


class Papers(Protocol):
    paper1_title: str
    paper2_title: str
    abstract_1: str
    abstract_2: str


@dataclass(frozen=True)
class Prompt:
    id: str
    system: str
    user: str  # str.format template

    @property
    def hash(self) -> str:
        blob = json.dumps(
            {"id": self.id, "system": self.system, "user": self.user},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def messages(self, **fields: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user.format(**fields)},
        ]


def render_papers(papers: Papers) -> str:
    return (
        "<papers>\n"
        f'<paper id="1">\nTitle: {papers.paper1_title}\n\n{papers.abstract_1.strip()}\n</paper>\n'
        f'<paper id="2">\nTitle: {papers.paper2_title}\n\n{papers.abstract_2.strip()}\n</paper>\n'
        "</papers>"
    )


def parse_json_object(text: str | None) -> dict[str, Any]:
    """A judge reply: one JSON object, optionally inside a single code block."""
    body, _ = unfence(text or "")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise JudgeOutputError(f"not JSON: {e}") from e
    if not isinstance(data, dict):
        raise JudgeOutputError(f"expected a JSON object, got {type(data).__name__}")
    return data


def parse_score(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise JudgeOutputError(f"{key!r} must be an integer 1-5, got {value!r}")
    return value
