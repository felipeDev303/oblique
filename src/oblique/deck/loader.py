"""Load and validate a deck directory: one card per YAML file, named after its id."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from oblique.deck.schema import Card, DeckError, validate_deck


def load_card(path: str | Path) -> Card:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: expected a mapping")
    card = Card.model_validate(data)
    if path.stem != card.id:
        raise ValueError(f"{path.name}: file name does not match id {card.id}")
    return card


def load_deck(directory: str | Path) -> dict[str, Card]:
    """All cards in `directory`, validated individually and as a deck."""
    cards: dict[str, Card] = {}
    problems: list[str] = []
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            card = load_card(path)
        except (ValidationError, ValueError) as e:
            problems.append(f"{path.name}: {e}")
            continue
        if card.id in cards:
            problems.append(f"{path.name}: duplicate id {card.id}")
            continue
        cards[card.id] = card
    if problems:
        raise DeckError(problems)
    validate_deck(cards)
    return cards
