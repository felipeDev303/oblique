"""Evolution G1 (ADR-010): one call to Muse Wonder, 15 proposed cards, no measurement.

    uv run python -m oblique.evolve.g1 configs/R-*.yaml --dry-run   # prints the prompt, no API call
    uv run python -m oblique.evolve.g1 configs/R-*.yaml

The call and its response are saved verbatim as the agent's record, in `record_dir`. Rerunning
reads the response cache, so it never pays twice. A proposal that breaks a rule is reported and
never re-asked: choosing among draws would put our judgement into the deck.

`validate_deck` only checks ids and opposites, so the rules of ADR-010 are checked here as well.
Two of them cannot be decided by code: "no technique" and "no paraphrase". For those the report
gives the closest existing card, for the author's review under ADR-002.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from pydantic import ValidationError

from oblique.deck import Card, DeckError, load_deck, validate_deck
from oblique.deck.controls import DECK_DIR
from oblique.prompts import EVOLVE_V1, JudgeOutputError
from oblique.prompts.base import parse_json_object
from oblique.smoke import Smoke, load_env

MIN_WORDS, MAX_WORDS = 5, 15
WONDER_TIMEOUT = 900.0  # seconds: long reasoning, and a timed-out attempt may still be billed
WONDER_ATTEMPTS = 2
# Above either of these, a card is flagged as possibly too close to an existing one.
CLOSE_CHARS, CLOSE_WORDS = 0.75, 0.6


def _ci(cell: dict[str, Any], signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{cell['mean']:{sign}.3f} [{cell['lo']:.3f}, {cell['hi']:.3f}]"


def render_table(rows: list[dict[str, Any]]) -> str:
    """Seed and paper cards in ID order: nothing in the order suggests which ones to follow."""
    lines = ["id | family | band | dq | dd | grounding | text"]
    for r in sorted(rows, key=lambda r: (r["origin"] != "seed", r["condition"])):
        if r["origin"] in ("seed", "paper"):
            lines.append(
                f"{r['condition']} | {r['family'] or '-'} | {r['band']} "
                f"| {_ci(r['q_controlled_minus_c2'], True)} | {_ci(r['d_minus_c2'], True)} "
                f"| {_ci(r['grounding'])} | {r['text']}"
            )
    return "\n".join(lines)


def render_controls(rows: list[dict[str, Any]]) -> str:
    """Aggregate only: the nouns themselves never reach the writer."""
    controls = [r for r in rows if r["origin"] == "control"]
    bands = {
        b: sum(r["band"] == b for r in controls) for b in sorted({r["band"] for r in controls})
    }
    counts = ", ".join(f"{n} {band}" for band, n in bands.items())

    def avg(key: str) -> float:
        return mean(r[key]["mean"] for r in controls)

    return (
        f"{len(controls)} cards, each a single random common noun with no strategy behind it: "
        f"{counts}; mean dq {avg('q_controlled_minus_c2'):+.3f}, "
        f"mean dd {avg('d_minus_c2'):+.3f}, mean grounding {avg('grounding'):.3f}."
    )


def evolve_messages(metrics: dict[str, Any], n: int) -> list[dict[str, str]]:
    rows = metrics["conditions"]
    ref = next(r for r in rows if r["origin"] == "no_card" and r["band"] == "reference")
    reference = f"grounding {_ci(ref['grounding'])}; by definition dq and dd are 0."
    return EVOLVE_V1.messages(
        reference=reference,
        table=render_table(rows),
        controls=render_controls(rows),
        n=str(n),
    )


def closest_card(text: str, deck: dict[str, Card]) -> dict[str, Any]:
    """The existing card most similar to `text`, by characters and by shared words."""

    def words(t: str) -> set[str]:
        return set(re.findall(r"[a-z']+", t.lower()))

    best: dict[str, Any] = {"id": None, "chars": 0.0, "words": 0.0}
    for card in deck.values():
        chars = SequenceMatcher(None, text.lower(), card.text.lower()).ratio()
        a, b = words(text), words(card.text)
        shared = len(a & b) / len(a | b) if a | b else 0.0
        if max(chars, shared) > max(best["chars"], best["words"]):
            best = {"id": card.id, "chars": round(chars, 3), "words": round(shared, 3)}
    return best


def validate_proposal(
    content: str | None, deck: dict[str, Card], n: int, author: str, created: date
) -> dict[str, Any]:
    """The proposal as evolved cards, plus every rule it breaks. Problems never drop a card."""
    problems: list[str] = []
    try:
        items = parse_json_object(content).get("cards")
    except JudgeOutputError as e:
        return {"cards": [], "problems": [f"response: {e}"], "close": []}
    if not isinstance(items, list):
        return {"cards": [], "problems": ["response: 'cards' is not a list"], "close": []}
    if len(items) != n:
        problems.append(f"expected {n} cards, got {len(items)}")

    allowed_parents = {c.id for c in deck.values() if c.origin in ("seed", "paper")}
    cards, built, close, seen = [], {}, [], {}
    for i, item in enumerate(items, 1):
        card_id = f"OB-G1-{i:03d}"
        if not isinstance(item, dict):
            problems.append(f"{card_id}: not an object")
            continue
        extra = set(item) - {"text", "family", "parents", "justification"}
        if extra:
            problems.append(f"{card_id}: unexpected fields {sorted(extra)}")
        text = item.get("text")
        text = text if isinstance(text, str) else ""
        parents = item.get("parents")
        parents = parents if isinstance(parents, list) else []
        justification = item.get("justification")
        justification = justification if isinstance(justification, str) else ""

        count = len(text.split())
        if not MIN_WORDS <= count <= MAX_WORDS:
            problems.append(f"{card_id}: {count} words, must be {MIN_WORDS}-{MAX_WORDS}")
        if text != text.strip() or not text.isascii():
            problems.append(f"{card_id}: text has outer spaces or non-ASCII characters")
        if not parents:
            problems.append(f"{card_id}: no parents")
        unknown = sorted(str(p) for p in parents if p not in allowed_parents)
        if unknown:
            problems.append(f"{card_id}: parents not in the table: {unknown}")
        if not justification.strip() or "\n" in justification.strip():
            problems.append(f"{card_id}: justification must be one line")
        key = " ".join(re.findall(r"[a-z']+", text.lower()))
        if key in seen:
            problems.append(f"{card_id}: same text as {seen[key]}")
        seen[key] = card_id

        try:
            card = Card(
                id=card_id,
                text=text,
                family=item.get("family"),
                origin="evolved",
                generation=1,
                parents=[str(p) for p in parents],
                author=author,
                created=created,
            )
            built[card_id] = card
        except ValidationError as e:
            problems.append(f"{card_id}: {e.errors()[0]['msg']}")

        nearest = closest_card(text, deck)
        if nearest["chars"] >= CLOSE_CHARS or nearest["words"] >= CLOSE_WORDS:
            close.append({"card": card_id, **nearest})
        cards.append(
            {
                "id": card_id,
                "text": text,
                "family": item.get("family"),
                "parents": parents,
                "justification": justification,
                "words": count,
                "closest": nearest,
            }
        )
    try:
        validate_deck({**deck, **built})
    except DeckError as e:
        problems += e.problems
    return {"cards": cards, "problems": problems, "close": close}


def render_summary(run_id: str, call: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [
        f"# {run_id} · Evolution G1 proposal (not measured)",
        "",
        f"One call to {call['model']}, US${call['cost_usd']:.4f}: {call['tokens_in']} tokens in, "
        f"{call['tokens_out']} out ({call['tokens_reasoning']} reasoning), "
        f"finish reason {call['finish_reason']}.",
        "",
        f"Rule problems: {'; '.join(report['problems']) or 'none'}.",
        f"Flagged as close to an existing card: "
        f"{', '.join(c['card'] for c in report['close']) or 'none'}.",
        "",
        "| Card | Family | Parents | Text | Justification |",
        "|---|---|---|---|---|",
    ]
    for c in report["cards"]:
        lines.append(
            f"| {c['id']} | {c['family']} | {', '.join(map(str, c['parents']))} "
            f"| {c['text']} | {c['justification']} |"
        )
    return "\n".join(lines) + "\n"


class Evolve(Smoke):
    def messages(self) -> list[dict[str, str]]:
        metrics = json.loads(Path(self.cfg["map_metrics"]).read_text("utf-8"))
        return evolve_messages(metrics, self.cfg["n_cards"])

    async def run(self) -> dict[str, Any]:
        cfg = self.cfg
        mode = cfg["writer"]
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")
        messages = self.messages()
        client = self.new_client(
            mode["provider"], timeout=WONDER_TIMEOUT, max_attempts=WONDER_ATTEMPTS
        )
        c = await self.call("evolve G1", mode, messages, client=client)
        call = self.calls[-1]

        record = Path(cfg["record_dir"])
        record.mkdir(parents=True, exist_ok=True)
        verbatim = {
            "run_id": self.run_id,
            "provider": mode["provider"],
            "model": mode["model"],
            "params": mode.get("params", {}),
            "prompt_id": EVOLVE_V1.id,
            "prompt_hash": EVOLVE_V1.hash,
            "call_key": c.key,
            "messages": messages,
            "response": c.response,
        }
        (record / "call.json").write_text(
            json.dumps(verbatim, indent=2, ensure_ascii=False, default=str) + "\n",
            "utf-8",
            newline="\n",
        )
        report = validate_proposal(
            c.content,
            {k: c for k, c in load_deck(cfg.get("deck", DECK_DIR)).items() if c.generation == 0},
            cfg["n_cards"],
            mode["model"],
            date.fromisoformat(str(cfg["created"])),
        )
        (record / "validation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
        )
        (self.out / "summary.md").write_text(
            render_summary(self.run_id, call, report), "utf-8", newline="\n"
        )
        (self.out / "calls.jsonl").write_text(
            json.dumps(call, default=str, ensure_ascii=False) + "\n", "utf-8", newline="\n"
        )
        return report


def main(argv: list[str]) -> None:
    if len(argv) not in (1, 2) or (len(argv) == 2 and argv[1] != "--dry-run"):
        raise SystemExit("usage: python -m oblique.evolve.g1 configs/R-*.yaml [--dry-run]")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    evolve = Evolve(cfg)
    if len(argv) == 2:
        for message in evolve.messages():
            print(f"--- {message['role']}\n{message['content']}\n")
        return
    load_env()
    report = asyncio.run(evolve.run())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
