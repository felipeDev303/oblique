"""Measure the G1 cards on split A (ADR-003, ADR-010 and its amendment).

    uv run python -m oblique.evolve.measure configs/R-*.yaml --write-cards   # the reviewed cards
    uv run python -m oblique.evolve.measure configs/R-*.yaml --plan          # counts, no API calls
    uv run --group embed python -m oblique.evolve.measure configs/R-*.yaml

One idea per reviewed card and instance, against the same anchors, S0 and C2 as the map: those come
from the response cache, and the run refuses to start if any of them would have to be generated.

The bands of G1 use a joint regression over the 48 conditions of the map and the G1 cards. The map
enters through its stored judgments, so nothing of it is asked again: a call the content filter
refused during the map is not retried, and the official map and the ML selection do not change.

MG: see `select_mg`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from oblique.bench import Instance
from oblique.cache import make_key
from oblique.deck import Card, load_deck, write_card
from oblique.deck.controls import DECK_DIR
from oblique.generate.schema import IdeaCard
from oblique.map_sweep import (
    CONDITION_SAMPLE_IDX,
    FALLBACK_SIZE,
    MIN_LIMINAL,
    REFERENCE,
    S0,
    MapSweep,
    render_table,
)
from oblique.prompts import ideation_messages, judged_words
from oblique.smoke import load_env

NOT_CONDITIONS = {S0, "ANCHOR"}


def reviewed_cards(cfg: dict[str, Any]) -> list[Card]:
    """The proposal minus the cards the author discarded, with the IDs and text of the proposal."""
    record = Path(cfg["record_dir"])
    report = json.loads((record / "validation.json").read_text("utf-8"))
    call = json.loads((record / "call.json").read_text("utf-8"))
    created = date.fromisoformat(str(cfg["created"]))
    return [
        Card(
            id=c["id"],
            text=c["text"],
            family=c["family"],
            origin="evolved",
            generation=1,
            parents=c["parents"],
            author=call["model"],
            created=created,
        )
        for c in report["cards"]
        if c["id"] not in cfg["discarded"]
    ]


def stored_regression_rows(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """One row per judgment of a finished map, with the rules of `MapSweep.regression_rows`:
    an invalid or filtered idea is two losses at equal length, and an idea missing one order of the
    judgment is left out."""
    run = Path(run_dir)

    def read(name: str) -> list[dict[str, Any]]:
        with (run / name).open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    anchors = {r["instance"]: r["idea"] for r in read("anchors.jsonl")}
    winners: dict[tuple[str, str], dict[bool, str]] = defaultdict(dict)
    for j in read("judgments.jsonl"):
        if j["judge"] == "quality-anchor" and j["parsed"] is not None:
            winners[(j["instance"], j["condition"])][j["idea_first"]] = j["parsed"]

    rows: list[dict[str, Any]] = []
    conditions: list[str] = []
    for g in read("generations.jsonl"):
        iid, cond = g["instance"], g["condition"]
        if cond in NOT_CONDITIONS:
            continue
        if cond not in conditions:
            conditions.append(cond)
        if g["idea"] is None:
            wins, delta = (False, False), 0
        else:
            verdict = winners.get((iid, cond), {})
            if len(verdict) != 2:
                continue
            wins = (verdict[True] == "A", verdict[False] == "B")
            delta = judged_words(IdeaCard(**g["idea"])) - judged_words(IdeaCard(**anchors[iid]))
        rows += [{"instance": iid, "condition": cond, "win": w, "delta_words": delta} for w in wins]
    return rows, conditions


def select_mg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """MG (ADR-010, amendment): every liminal G1 card; with fewer than 3, the 6 non-destructive ones
    with the highest m (or all, if fewer), marked as fallback. With none, MG leaves split B and H4
    is not tested."""
    evolved = [r for r in rows if r["origin"] == "evolved"]
    liminal = sorted(r["condition"] for r in evolved if r["band"] == "liminal")
    if len(liminal) >= MIN_LIMINAL:
        return {"cards": liminal, "fallback": False, "liminal": liminal, "h4_testable": True}
    candidates = [r for r in evolved if r["band"] != "destructive" and r["m"] is not None]
    candidates.sort(key=lambda r: (-r["m"]["mean"], r["condition"]))
    cards = [r["condition"] for r in candidates[:FALLBACK_SIZE]]
    return {"cards": cards, "fallback": True, "liminal": liminal, "h4_testable": bool(cards)}


class MeasureG1(MapSweep):
    def __init__(self, cfg: dict[str, Any], **kw: Any):
        super().__init__(cfg, **kw)
        self.map_rows, self.map_conditions = stored_regression_rows(cfg["map_run"])
        expected = json.loads((Path(cfg["map_run"]) / "metrics.json").read_text("utf-8"))
        self.map_judgments = expected["length"]["judgments_in_regression"]
        if len(self.map_rows) != self.map_judgments:
            raise SystemExit(
                f"stored rows {len(self.map_rows)} differ from the map's {self.map_judgments}"
            )

    def check_cards(self) -> None:
        """The config measures every reviewed card, and the deck holds exactly those."""
        reviewed = sorted(c.id for c in reviewed_cards(self.cfg))
        in_deck = sorted(c.id for c in self.deck.values() if c.origin == "evolved")
        if sorted(self.cfg["cards"]["evolved"]) != reviewed or in_deck != reviewed:
            raise SystemExit(f"config and deck must hold the reviewed cards: {reviewed}")
        for card in reviewed_cards(self.cfg):
            if self.deck[card.id].text != card.text:
                raise SystemExit(f"{card.id}: the text differs from the proposal")

    def check_reference_cached(self, instances: list[Instance]) -> None:
        """Same anchors, S0 and C2 as the map, or nothing runs."""
        blocks = self.plan(instances)["blocks"]
        gen = self.cfg["generator"]
        ideation = self.prompt("ideation")
        c2_keys = [
            make_key(
                gen["model"],
                ideation_messages(x, 1, ideation),
                {**gen["params"], **self.no_card[REFERENCE]},
                CONDITION_SAMPLE_IDX,
                0,
                provider=gen["provider"],
            )
            for x in instances
        ]
        missing = {
            "anchor": blocks["anchor"]["to_pay"],
            "s0": blocks["s0"]["to_pay"],
            "c2": sum(key not in self.cache for key in c2_keys),
        }
        if any(missing.values()):
            raise SystemExit(f"not the map's anchors, S0 or C2; would generate {missing}")

    def regression_extra(self) -> tuple[list[dict[str, Any]], list[str]]:
        rows = [r for r in self.map_rows if r["condition"] != REFERENCE]
        return rows, [c for c in self.map_conditions if c != REFERENCE]

    def report(self, instances: list[Instance]) -> dict[str, Any]:
        result = super().report(instances)
        del result["ml"]
        result["mg"] = select_mg(result["conditions"])
        stored_c2 = sorted(
            (r["instance"], r["win"], r["delta_words"])
            for r in self.map_rows
            if r["condition"] == REFERENCE
        )
        result["c2_same_as_map"] = stored_c2 == sorted(
            (r["instance"], r["win"], r["delta_words"])
            for r in self.current_rows
            if r["condition"] == REFERENCE
        )
        result["regression"] = {
            "conditions": len(self.map_conditions) + len(self.cards),
            "map_rows": self.map_judgments,
        }
        return result

    def write_summary(self, result: dict[str, Any]) -> None:
        (self.out / "summary.md").write_text(render_g1_summary(result), "utf-8", newline="\n")


def render_g1_summary(result: dict[str, Any]) -> str:
    mg = result["mg"]
    kind = "fallback" if mg["fallback"] else "liminal G1 cards"
    return "\n".join(
        [
            f"# {result['run_id']} · The G1 cards on split A",
            "",
            f"{result['instances']} instances, {len(result['conditions']) - 1} cards, "
            f"US${result['spent_usd']:.2f}. Bands from a joint regression over "
            f"{result['regression']['conditions']} conditions (ADR-010, amendment). "
            f"Same C2 judgments as the map: {result['c2_same_as_map']}.",
            "",
            f"**MG ({kind}):** {', '.join(mg['cards']) or 'none: H4 is not tested'}.",
            "",
            f"**Liminal confirmed:** {', '.join(result['liminal_confirmed']) or 'none'}.",
            "",
            *render_table(result),
        ]
    )


def main(argv: list[str]) -> None:
    flags = ("--plan", "--write-cards")
    if len(argv) not in (1, 2) or (len(argv) == 2 and argv[1] not in flags):
        raise SystemExit("usage: python -m oblique.evolve.measure configs/R-*.yaml [flag]")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    if argv[1:] == ["--write-cards"]:
        for card in reviewed_cards(cfg):
            print(write_card(card, cfg.get("deck", DECK_DIR)))
        load_deck(cfg.get("deck", DECK_DIR))
        return
    measure = MeasureG1(cfg)
    if argv[1:] == ["--plan"]:
        measure.check_cards()
        print(json.dumps(measure.plan(measure.instances()), indent=2))
        return
    load_env()
    asyncio.run(measure.run())


if __name__ == "__main__":
    main(sys.argv[1:])
