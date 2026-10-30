"""Taxonomy label conformity (ADR-006, sections 2 and 3).

`problem_gap` should open with an opportunity pattern and `method` with a
method paradigm, in square brackets. Conformity never decides whether an idea
is judged; it decides whether the label enters the type-entropy metrics and
feeds the strict-rule sensitivity analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

OPPORTUNITY_PATTERNS = (
    "Puzzle / Contradiction",
    "Explanation Gap",
    "Scope Mismatch",
    "Evidence Gap",
    "Bridge Opportunity",
    "Failure / Risk Gap",
    "Resource Bottleneck",
    "Other",
)

METHOD_PARADIGMS = (
    "Synthesis / Unification",
    "Relax / Extend Scope",
    "Robustification",
    "Formal Derivation",
    "Empirical Mapping",
    "Artifact / System",
    "Optimization / Search",
    "Other",
)

OTHER = "Other"
_LEADING_LABEL = re.compile(r"^\s*\[([^\]]+)\]")
_SLASH = re.compile(r"\s*/\s*")


class LabelStatus(StrEnum):
    ok = "ok"  # matches a canonical label
    other = "other"  # [Other]
    off_list = "off_list"  # bracketed, but not in the list
    missing = "missing"  # no leading bracketed label


CONFORMING = frozenset({LabelStatus.ok, LabelStatus.other})


def normalize_label(label: str) -> str:
    """Lowercase, collapse whitespace, drop spaces around '/'."""
    collapsed = " ".join(label.lower().split())
    return _SLASH.sub("/", collapsed)


@dataclass(frozen=True)
class LabelResult:
    status: LabelStatus
    canonical: str | None  # canonical spelling when ok/other
    raw: str | None  # text inside the brackets, as written

    @property
    def conforming(self) -> bool:
        return self.status in CONFORMING


def _index(labels: tuple[str, ...]) -> dict[str, str]:
    return {normalize_label(label): label for label in labels}


_INDEX = {
    OPPORTUNITY_PATTERNS: _index(OPPORTUNITY_PATTERNS),
    METHOD_PARADIGMS: _index(METHOD_PARADIGMS),
}


def classify_label(field_text: str, labels: tuple[str, ...]) -> LabelResult:
    """Read the leading [label] of a field and compare it with a canonical list."""
    match = _LEADING_LABEL.match(field_text)
    if match is None:
        return LabelResult(LabelStatus.missing, None, None)
    raw = match.group(1)
    key = normalize_label(raw)
    if not key:
        return LabelResult(LabelStatus.missing, None, raw)
    index = _INDEX.get(labels) or _index(labels)
    canonical = index.get(key)
    if canonical is None:
        return LabelResult(LabelStatus.off_list, None, raw)
    status = LabelStatus.other if canonical == OTHER else LabelStatus.ok
    return LabelResult(status, canonical, raw)


def classify_opportunity(problem_gap: str) -> LabelResult:
    return classify_label(problem_gap, OPPORTUNITY_PATTERNS)


def classify_method(method: str) -> LabelResult:
    return classify_label(method, METHOD_PARADIGMS)
