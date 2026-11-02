"""Measuring G1 against a small fake map: same reference, joint regression, MG. No network."""

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from tenacity import wait_none
from test_map_sweep import INSTANCES, FakeAPI, FakeEmbedder

from oblique import baseline, map_sweep
from oblique.deck import load_deck
from oblique.deck.controls import DECK_DIR
from oblique.evolve.measure import MeasureG1, reviewed_cards, select_mg, stored_regression_rows
from oblique.map_sweep import MapSweep

REAL_CONFIG = "configs/R-20261102-01.yaml"
CARDS = ["OB-G1-001", "OB-G1-002"]


def _cell(value):
    return {"mean": value, "lo": value, "hi": value, "n": 50}


def _row(card, band, m=0.1):
    return {"condition": card, "origin": "evolved", "band": band, "m": _cell(m)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    monkeypatch.setattr(map_sweep, "load_instances", lambda split: INSTANCES)
    with open("configs/R-20261031-01.yaml", encoding="utf-8") as f:
        map_cfg = yaml.safe_load(f)
    map_cfg |= {
        "cache": str(tmp_path / "cache.db"),
        "cards": {"seed": ["OB-001"], "paper": ["E6-1"], "control": ["CTRL-R-001"]},
        "n_boot": 50,
    }
    api = FakeAPI()
    first = MapSweep(
        map_cfg, out=tmp_path / "map", transport=api, wait=wait_none(), embedder=FakeEmbedder()
    )
    first.check_cards = lambda: None
    asyncio.run(first.run())
    with open(REAL_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg |= {
        "cache": str(tmp_path / "cache.db"),
        "map_run": str(tmp_path / "map"),
        "cards": {"evolved": CARDS},
        "n_boot": 50,
    }
    return {"api": api, "cfg": cfg, "map": first, "out": tmp_path / "g1", "tmp": tmp_path}


def measure(world, api=None):
    run = MeasureG1(
        world["cfg"],
        out=world["out"],
        transport=api or world["api"],
        wait=wait_none(),
        embedder=FakeEmbedder(),
    )
    run.check_cards = lambda: None
    return run


def test_stored_rows_rebuild_the_regression_of_the_map(world):
    rows, conditions = stored_regression_rows(world["tmp"] / "map")
    assert sorted(conditions) == sorted(world["map"].conditions)
    key = lambda r: (r["instance"], r["condition"], r["win"], r["delta_words"])  # noqa: E731
    assert sorted(map(key, rows)) == sorted(map(key, world["map"].current_rows))


def test_only_the_new_cards_are_generated_and_the_reference_comes_from_the_cache(world):
    before = len(world["api"].requests)
    run = measure(world)
    asyncio.run(run.run())
    new = world["api"].requests[before:]
    generations = [r for r in new if "research scientist" in r["messages"][0]["content"]]
    assert len(generations) == len(CARDS) * len(INSTANCES)
    assert all("oblique constraint" in r["messages"][1]["content"] for r in generations)
    # C2's judgments are cached too: every paid judge call reads an idea under a new card.
    assert len(new) == len(CARDS) * len(INSTANCES) * (1 + 2 + 1 + 3)


def test_the_regression_is_joint_but_only_g1_is_reported(world):
    map_metrics = (world["tmp"] / "map" / "metrics.json").read_text("utf-8")
    asyncio.run(measure(world).run())
    result = json.loads((world["out"] / "metrics.json").read_text("utf-8"))
    assert [r["condition"] for r in result["conditions"]] == [*CARDS, "C2"]
    assert result["regression"]["conditions"] == len(world["map"].conditions) + len(CARDS)
    assert result["c2_same_as_map"] is True
    assert result["length"]["judgments_in_regression"] == (
        len(stored_regression_rows(world["tmp"] / "map")[0]) + 2 * len(CARDS) * len(INSTANCES)
    )
    assert "ml" not in result and "mg" in result
    assert "MG" in (world["out"] / "summary.md").read_text("utf-8")
    assert (world["tmp"] / "map" / "metrics.json").read_text("utf-8") == map_metrics


def test_it_refuses_to_run_without_the_reference_of_the_map(world, tmp_path):
    world["cfg"]["cache"] = str(tmp_path / "empty.db")
    api = FakeAPI()
    with pytest.raises(SystemExit, match="would generate"):
        asyncio.run(measure(world, api).run())
    assert api.requests == []


def test_a_resume_passes_the_check_and_pays_nothing(world):
    first = measure(world)
    asyncio.run(first.run())
    paid = len(world["api"].requests)
    again = measure(world)
    asyncio.run(again.run())
    assert len(world["api"].requests) == paid
    assert again.spent_usd == pytest.approx(first.spent_usd)


def test_mg_is_every_liminal_card():
    rows = [_row("B", "liminal"), _row("A", "liminal"), _row("C", "liminal"), _row("D", "inert")]
    assert select_mg(rows) == {
        "cards": ["A", "B", "C"],
        "fallback": False,
        "liminal": ["A", "B", "C"],
        "h4_testable": True,
    }


def test_mg_falls_back_to_the_six_non_destructive_cards_with_the_highest_m():
    rows = [_row("D0", "destructive", 0.9), _row("L0", "liminal", 0.01)]
    rows += [_row(f"I{i}", "inert", 0.1 * i) for i in range(1, 8)]
    mg = select_mg(rows)
    assert mg["fallback"] is True and mg["h4_testable"] is True
    assert mg["cards"] == ["I7", "I6", "I5", "I4", "I3", "I2"]
    assert mg["liminal"] == ["L0"]


def test_mg_falls_back_to_all_when_fewer_than_six_and_h4_falls_with_none():
    few = select_mg([_row("A", "inert", 0.2), _row("B", "destructive")])
    assert few["cards"] == ["A"] and few["fallback"] is True
    none = select_mg([_row("A", "destructive")])
    assert none["cards"] == [] and none["h4_testable"] is False


def test_the_committed_config_measures_the_reviewed_cards_as_written():
    with open(REAL_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cards = reviewed_cards(cfg)
    assert len(cards) == 13
    assert {"OB-G1-011", "OB-G1-014"} == set(cfg["discarded"])
    assert sorted(c.id for c in cards) == sorted(cfg["cards"]["evolved"])
    deck = load_deck(DECK_DIR)
    assert all(deck[c.id] == c for c in cards)
    response = json.loads(Path("deck/evolution/G1/call.json").read_text("utf-8"))["response"]
    proposed = json.loads(response["choices"][0]["message"]["content"])["cards"]
    assert {c.text for c in cards} <= {p["text"] for p in proposed}
