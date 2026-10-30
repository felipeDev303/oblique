"""Random-word control cards CTRL-R-001..030 (condition R).

    uv run --group deck python -m oblique.deck.controls

Source: the noun list shipped with the `wonderwords` package (2.2.0,
`assets/nounlist.txt`, MIT). Method, fixed before any result:
1. keep lowercase, purely alphabetic entries;
2. drop entries in the package's own profanity list;
3. keep common words: English Zipf frequency >= 4.0 in `wordfreq` 3.1.1
   (about 10 or more occurrences per million words);
4. sort the pool, then draw 30 words without replacement with the `cards`
   seed; the i-th word drawn becomes CTRL-R-{i:03d}.
Both packages are pinned in the `deck` dependency group: another version of
either list could change the pool. Rerunning rewrites the same 30 files.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from oblique.deck.loader import write_card
from oblique.deck.schema import Card
from oblique.seeds import rng

N_CONTROLS = 30
MIN_ZIPF = 4.0
CREATED = date(2026, 10, 30)
DECK_DIR = Path(__file__).resolve().parents[3] / "deck" / "cards"


def common_nouns(min_zipf: float = MIN_ZIPF) -> list[str]:
    import importlib.resources

    from wordfreq import zipf_frequency

    assets = importlib.resources.files("wonderwords") / "assets"
    nouns = set((assets / "nounlist.txt").read_text(encoding="utf-8").split())
    profane = set((assets / "profanitylist.txt").read_text(encoding="utf-8").split())
    return sorted(
        w
        for w in nouns
        if w.isalpha() and w.islower() and w not in profane and zipf_frequency(w, "en") >= min_zipf
    )


def draw_words(pool: Sequence[str], n: int = N_CONTROLS) -> list[str]:
    """n distinct words from the sorted pool, in draw order, from the `cards` seed."""
    pool = sorted(set(pool))
    if len(pool) < n:
        raise ValueError(f"pool has {len(pool)} words, need {n}")
    return [pool[i] for i in rng("cards").choice(len(pool), size=n, replace=False)]


def control_cards(words: Sequence[str], created: date = CREATED) -> list[Card]:
    return [
        Card(
            id=f"CTRL-R-{i:03d}",
            text=word,
            family=None,
            origin="control",
            author="random",
            created=created,
        )
        for i, word in enumerate(words, start=1)
    ]


def main(argv: Sequence[str]) -> None:
    directory = Path(argv[0]) if argv else DECK_DIR
    pool = common_nouns()
    for card in control_cards(draw_words(pool)):
        write_card(card, directory)
        print(card.id, card.text)
    print(f"pool: {len(pool)} nouns; wrote {N_CONTROLS} controls to {directory}")


if __name__ == "__main__":
    main(sys.argv[1:])
