"""The committed deck, the card writer and the control-card draw. No network needed;
the check against the documented method runs only where the `deck` group is installed."""

from collections import Counter
from datetime import date

import pytest

from oblique.deck import Card, Family, dump_card, load_card, load_deck, write_card
from oblique.deck.controls import DECK_DIR, N_CONTROLS, control_cards, draw_words, pick_controls


def test_committed_deck_loads_and_has_the_planned_composition():
    deck = load_deck(DECK_DIR)
    origins = Counter(c.origin for c in deck.values())
    assert origins == {"seed": 30, "paper": 6, "control": 30}
    families = Counter(c.family for c in deck.values() if c.origin == "seed")
    assert set(families) == set(Family) and min(families.values()) >= 3
    pairs = {frozenset((c.id, o)) for c in deck.values() for o in c.opposite_of}
    assert len(pairs) >= 4
    assert all(c.family is None for c in deck.values() if c.origin in ("paper", "control"))
    assert all(c.generation == 0 for c in deck.values())


def test_control_texts_are_single_distinct_words():
    controls = [c for c in load_deck(DECK_DIR).values() if c.origin == "control"]
    texts = [c.text for c in controls]
    assert len(set(texts)) == N_CONTROLS
    assert all(t.isalpha() and t.islower() for t in texts)
    assert all(c.author == "random" for c in controls)


@pytest.mark.parametrize(
    "text", ['Say "no": then stop.', "Read it as #1 - or not?", "Ünïcode — ok"]
)
def test_dump_card_round_trips_text_exactly(tmp_path, text):
    card = Card(
        id="OB-099",
        text=text,
        family="error",
        origin="seed",
        opposite_of=["OB-098"],
        author="felipe",
        created=date(2026, 10, 30),
    )
    assert load_card(write_card(card, tmp_path)) == card
    assert dump_card(card).splitlines()[0] == "id: OB-099"


def test_draw_words_is_reproducible_distinct_and_order_independent():
    pool = [f"word{i}" for i in range(500)]
    a = draw_words(pool)
    assert a == draw_words(list(reversed(pool)))
    assert len(a) == len(set(a)) == N_CONTROLS and set(a) <= set(pool)
    with pytest.raises(ValueError):
        draw_words(pool[:10])


def test_control_cards_follow_the_schema():
    cards = control_cards(["alpha", "beta"])
    assert [c.id for c in cards] == ["CTRL-R-001", "CTRL-R-002"]
    assert all(c.origin == "control" and c.family is None for c in cards)


def test_committed_controls_match_the_documented_method():
    controls = pytest.importorskip("oblique.deck.controls")
    pytest.importorskip("wonderwords")
    pytest.importorskip("wordfreq")
    deck = load_deck(DECK_DIR)
    expected = draw_words(controls.common_nouns())
    assert [deck[f"CTRL-R-{i:03d}"].text for i in range(1, N_CONTROLS + 1)] == expected


def test_pick_controls_is_seeded_and_order_independent():
    ids = [f"CTRL-R-{i:03d}" for i in range(1, 31)]
    picked = pick_controls(ids, 10)
    assert len(set(picked)) == 10 and set(picked) <= set(ids) and picked == sorted(picked)
    assert picked == pick_controls(list(reversed(ids)), 10)
