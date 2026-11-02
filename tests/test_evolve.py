"""Evolution G1: the prompt, the verbatim record and the rule checks. No API calls."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import pytest

from oblique.deck import load_deck
from oblique.deck.controls import DECK_DIR
from oblique.evolve.g1 import Evolve, closest_card, evolve_messages, validate_proposal

DECK = load_deck(DECK_DIR)
CREATED = date(2026, 11, 1)


def _cell(value):
    return {"mean": value, "lo": value - 0.1, "hi": value + 0.1, "n": 50}


def _row(condition, origin, band="liminal", text="A card text."):
    return {
        "condition": condition,
        "origin": origin,
        "family": "error" if origin == "seed" else None,
        "band": band,
        "text": text,
        "q_controlled_minus_c2": None if origin == "no_card" else _cell(-0.02),
        "d_minus_c2": None if origin == "no_card" else _cell(0.1),
        "grounding": _cell(0.7),
    }


METRICS = {
    "conditions": [
        _row("OB-002", "seed", text="Second seed."),
        _row("OB-001", "seed", "destructive", text="First seed."),
        _row("E6-1", "paper", text="Paper opening."),
        _row("CTRL-R-001", "control", "inert", text="zebra"),
        _row("CTRL-R-002", "control", text="lantern"),
        _row("C2", "no_card", "reference", text=None),
        _row("C2-T", "no_card", "inert", text=None),
    ]
}


def _item(text, family="inversion", parents=("OB-001",), justification="Because it may help."):
    return {
        "text": text,
        "family": family,
        "parents": list(parents),
        "justification": justification,
    }


def _proposal(*items):
    return json.dumps({"cards": list(items)})


def _check(content, n=1):
    return validate_proposal(content, DECK, n, "muse-wonder-1.0", CREATED)


def test_prompt_has_seed_and_paper_cards_and_only_an_aggregate_of_the_nouns():
    system, user = (m["content"] for m in evolve_messages(METRICS, 15))
    assert "First seed." in user and "Second seed." in user and "Paper opening." in user
    assert "zebra" not in user and "lantern" not in user
    assert "CTRL-R" not in user
    assert "2 cards, each a single random common noun" in user
    assert "1 inert, 1 liminal" in user
    assert "exactly 15 objects" in user
    assert "subtraction, inversion, scale, role, constraint, ambiguity_specificity" in system
    assert user.index("E6-1") > user.index("OB-002") > user.index("OB-001")


def test_a_valid_proposal_builds_evolved_cards_without_problems():
    report = _check(
        _proposal(_item("Trace the disagreement back to whoever collected the samples."))
    )
    assert report["problems"] == []
    assert report["cards"][0]["id"] == "OB-G1-001"
    assert report["cards"][0]["words"] == 9


@pytest.mark.parametrize(
    "item, expected",
    [
        (_item("Too short here."), "3 words"),
        (_item(" ".join(["word"] * 16)), "16 words"),
        (_item("A perfectly valid length of card.", family="magic"), "OB-G1-001"),
        (_item("A perfectly valid length of card.", parents=()), "no parents"),
        (_item("A perfectly valid length of card.", parents=("OB-999",)), "not in the table"),
        (_item("A perfectly valid length of card.", parents=("CTRL-R-001",)), "not in the table"),
        (_item("A perfectly valid length of card.", justification=""), "one line"),
        (_item("A perfectly valid length of card.", justification="a\nb"), "one line"),
        (_item("Una carta con acentos también aquí."), "non-ASCII"),
    ],
)
def test_each_broken_rule_is_reported(item, expected):
    assert any(expected in p for p in _check(_proposal(item))["problems"])


def test_wrong_count_duplicates_and_bad_json_are_reported_and_nothing_is_dropped():
    text = "A perfectly valid length of card."
    report = _check(_proposal(_item(text), _item(text)), n=15)
    assert any("expected 15 cards, got 2" in p for p in report["problems"])
    assert any("same text as OB-G1-001" in p for p in report["problems"])
    assert len(report["cards"]) == 2
    assert _check("not json")["problems"][0].startswith("response:")
    assert _check('{"cards": 3}')["problems"] == ["response: 'cards' is not a list"]


def test_a_copy_of_an_existing_card_is_flagged():
    existing = DECK["OB-001"].text
    assert closest_card(existing, DECK)["id"] == "OB-001"
    report = _check(_proposal(_item(existing)))
    assert report["close"][0]["card"] == "OB-G1-001"
    assert report["close"][0]["id"] == "OB-001"


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeWonder:
    def __init__(self):
        self.chat = self.completions = self
        self.models = SimpleNamespace()
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        content = _proposal(_item("Trace the disagreement back to whoever collected the samples."))
        return _Raw(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            }
        )


@pytest.fixture
def cfg(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(METRICS), "utf-8")
    return {
        "run_id": "R-TEST",
        "cache": str(tmp_path / "cache.db"),
        "map_metrics": str(metrics),
        "n_cards": 1,
        "created": "2026-11-01",
        "record_dir": str(tmp_path / "record"),
        "writer": {"provider": "meta", "model": "muse-wonder-1.0", "params": {"max_tokens": 100}},
        "budget": {
            "stop_usd": 0.5,
            "ledger": True,
            "prices_per_m": {"muse-wonder-1.0": {"in": 4.0, "cached": 0.2, "out": 20.0}},
        },
    }


def test_the_call_is_saved_verbatim_and_a_rerun_pays_nothing(cfg, tmp_path):
    api = FakeWonder()
    first = Evolve(cfg, out=tmp_path / "run", transport=api)
    report = asyncio.run(first.run())
    assert report["problems"] == []
    assert first.spent_usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)

    saved = json.loads((tmp_path / "record" / "call.json").read_text("utf-8"))
    assert saved["messages"] == api.requests[0]["messages"]
    assert saved["response"]["choices"][0]["message"]["content"] == _proposal(
        _item("Trace the disagreement back to whoever collected the samples.")
    )
    assert saved["model"] == "muse-wonder-1.0"
    assert "OB-G1-001" in (tmp_path / "run" / "summary.md").read_text("utf-8")

    again = Evolve(cfg, out=tmp_path / "run", transport=api)
    asyncio.run(again.run())
    assert len(api.requests) == 1
    assert again.spent_usd == pytest.approx(first.spent_usd)
