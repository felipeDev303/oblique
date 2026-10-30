"""Deck card schema (`deck/cards/*.yaml`).

Seed and evolved cards carry one of eight strategy families. Control cards
(random words) and paper cards (the six openings, condition E6) carry no
family: `origin` already separates them, and classifying a foreign deck into
our families would put our judgement into the comparison.

`opposite_of` is symmetric across the deck: if A lists B, B lists A.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CARD_ID = r"^(OB-(G\d+-)?\d{3}|CTRL-R-\d{3}|E6-[1-6])$"
_EVOLVED_ID = re.compile(r"^OB-G(\d+)-\d{3}$")
_ID_PREFIX = {
    "seed": re.compile(r"^OB-\d{3}$"),
    "evolved": _EVOLVED_ID,
    "control": re.compile(r"^CTRL-R-\d{3}$"),
    "paper": re.compile(r"^E6-[1-6]$"),
}


class Family(StrEnum):
    subtraction = "subtraction"
    inversion = "inversion"
    scale = "scale"
    role = "role"
    constraint = "constraint"
    ambiguity_specificity = "ambiguity_specificity"
    error = "error"
    time = "time"


Origin = Literal["seed", "evolved", "control", "paper"]


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=CARD_ID)
    text: str = Field(min_length=1)
    family: Family | None = None
    origin: Origin
    generation: int = Field(default=0, ge=0)
    parents: list[str] = []
    opposite_of: list[str] = []
    author: str = Field(min_length=1)
    created: date

    @model_validator(mode="after")
    def _check(self) -> Card:
        needs_family = self.origin in ("seed", "evolved")
        if needs_family and self.family is None:
            raise ValueError(f"{self.id}: family required for origin={self.origin}")
        if not needs_family and self.family is not None:
            raise ValueError(f"{self.id}: family must be null for origin={self.origin}")
        if self.origin != "evolved" and self.parents:
            raise ValueError(f"{self.id}: only evolved cards have parents")
        if not self.text.strip():
            raise ValueError(f"{self.id}: text is blank")

        # Consistency with the ID conventions.
        if not _ID_PREFIX[self.origin].match(self.id):
            raise ValueError(f"{self.id}: id does not match origin={self.origin}")
        evolved = _EVOLVED_ID.match(self.id)
        expected_gen = int(evolved.group(1)) if evolved else 0
        if self.generation != expected_gen:
            raise ValueError(f"{self.id}: generation must be {expected_gen}")

        if self.id in self.opposite_of:
            raise ValueError(f"{self.id}: a card cannot be its own opposite")
        if len(set(self.opposite_of)) != len(self.opposite_of):
            raise ValueError(f"{self.id}: duplicate ids in opposite_of")
        return self


class DeckError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid deck:\n  " + "\n  ".join(problems))


def validate_deck(cards: dict[str, Card]) -> None:
    """Cross-card checks: keys match ids, opposites exist and are reciprocal.

    An evolved card may declare an opposite only if that card already exists in
    the deck (and then that card must list it back).
    """
    problems: list[str] = []
    for key, card in cards.items():
        if key != card.id:
            problems.append(f"key {key} holds card {card.id}")
        for other in card.opposite_of:
            if other not in cards:
                problems.append(f"{card.id}: opposite {other} not in deck")
            elif card.id not in cards[other].opposite_of:
                problems.append(f"{card.id} <-> {other}: opposite_of must be symmetric")
    if problems:
        raise DeckError(problems)
