"""Idea card schema: the four fields every generated idea must carry.

Field text is kept exactly as generated (no stripping, no label removal), so
judges see what the model wrote (ADR-006).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator

IDEA_FIELDS = ("problem_gap", "core_claim", "method", "inspiration")


class IdeaCard(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    problem_gap: StrictStr
    core_claim: StrictStr
    method: StrictStr
    inspiration: StrictStr

    @field_validator(*IDEA_FIELDS)
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field is empty or whitespace only")
        return v
