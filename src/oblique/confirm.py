"""The confirmation on split B, once (ADR-003, ADR-009, ADR-010 and its amendment 2).

    uv run python -m oblique.confirm configs/R-*.yaml --write-draws   # card draws, from IDs only
    uv run python -m oblique.confirm configs/R-*.yaml --plan          # the gate: no API, B sealed
    OBLIQUE_UNSEAL_B=1 uv run --group embed python -m oblique.confirm configs/R-*.yaml

`--plan` and `--write-draws` read only the instance IDs of `splits/B.txt`, which are not sealed
content. The card of every idea is drawn from those IDs and committed before B is opened.

Sets of K = 3 written in sequence (ADR-009): C1s, Ts, R, E6, ML, MG. C1 is one call, the reference
of the paper. Every idea is judged against a new anchor of split B in both orders, for grounding,
and for distance to the other ideas of its set.

One run only (amendment 2). A finished run has `metrics.json` and refuses to start again. A run that
was cut resumes with the same command: finished calls come from the response cache, and a call the
content filter refused is remembered in `filtered.jsonl` and never asked again, so nothing that
finished runs twice.

q controlled for length comes from a regression on the judgments of split B only, refitted on every
resample. CU uses raw q (ADR-009). Comparisons are paired on the instances complete in both
conditions (ADR-006, amendment 6). The report opens with H1, then the grounding guard, then the
exploratory hypotheses.

Agreement (amendment 3): Muse Wonder judges 30 ideas per condition against the anchor, in both
orders. The same drawn positions (instance, slot) are used in every condition, so its difference
between ML and C1s is paired. It is a sensitivity analysis and does not decide H1.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oblique.baseline import SEQUENTIAL_SAMPLE_BASE, Baseline, K, ci_dict
from oblique.bench import Instance, load_instances, split_ids
from oblique.deck import Card, load_deck
from oblique.deck.controls import DECK_DIR
from oblique.generate.records import records_from_set, write_jsonl
from oblique.generate.validate import InvalidReason, provider_error_set
from oblique.llm import Usage
from oblique.metrics import paired_differences
from oblique.metrics.bootstrap import bca_mean_ci, bootstrap_mean_ci
from oblique.metrics.calibration import (
    agreement,
    controlled_win_rates,
    fit_condition_logit,
    spearman,
)
from oblique.prompts import (
    judged_words,
    parse_winner,
    q_from_orders,
    quality_messages,
    render_card,
)
from oblique.seeds import rng, seed_sequence
from oblique.smoke import load_env, write_jsonl_dicts

DECK_CONDITIONS = ("R", "E6", "ML", "MG")  # the order is part of the pre-registered draw
SPLIT = "B"
BASELINE = "C1s"
PRIMARY = "ML"


def draw_cards(
    instance_ids: list[str], decks: dict[str, list[str]], k: int = K
) -> dict[str, dict[str, list[str]]]:
    """The K cards of each set, without replacement inside a set, with the `cards` seed. Each deck
    has its own stream, so no deck's draw depends on the size of another."""
    streams = seed_sequence("cards").spawn(len(DECK_CONDITIONS))
    draws: dict[str, dict[str, list[str]]] = {}
    for condition, stream in zip(DECK_CONDITIONS, streams, strict=True):
        generator = np.random.default_rng(stream)
        pool = sorted(decks[condition])
        draws[condition] = {
            iid: [pool[i] for i in generator.choice(len(pool), size=k, replace=False)]
            for iid in sorted(instance_ids)
        }
    return draws


def draw_wonder_positions(instance_ids: list[str], n: int, k: int = K) -> list[list[Any]]:
    """The (instance, slot) positions Wonder judges in every condition, with the `wonder_b` seed."""
    positions = [(iid, slot) for iid in sorted(instance_ids) for slot in range(k)]
    chosen = rng("wonder_b").choice(len(positions), size=n, replace=False)
    return [list(positions[i]) for i in sorted(chosen.tolist())]


def h1_verdict(delta_cu: dict[str, float], delta_q: dict[str, float], delta: float) -> str:
    """ADR-003. Supported: CI of the CU difference above 0 and lower bound of q above -delta.
    Refuted: CI of the CU difference below 0, or upper bound of q below -delta."""
    if delta_cu["hi"] < 0 or delta_q["hi"] < -delta:
        return "refuted"
    if delta_cu["lo"] > 0 and delta_q["lo"] > -delta:
        return "supported"
    return "inconclusive"


def bootstrap_seconds(rows: int, conditions: int, fits: int = 20) -> float:
    """Seconds per regression fit, on synthetic data of the size of the run."""
    generator = np.random.default_rng(0)
    condition = generator.integers(0, conditions, rows)
    x = generator.normal(0, 1, rows)
    y = (generator.random(rows) < 0.5).astype(float)
    started = time.perf_counter()
    for _ in range(fits):
        fit_condition_logit(condition, x, y, conditions)
    return (time.perf_counter() - started) / fits


class ConfirmB(Baseline):
    def __init__(self, cfg: dict[str, Any], **kw: Any):
        super().__init__(cfg, **kw)
        self.deck: dict[str, Card] = load_deck(cfg.get("deck", DECK_DIR))
        self.decks: dict[str, list[str]] = cfg["decks"]
        self.no_card: dict[str, dict[str, Any]] = cfg["no_card"]
        self.ids = sorted(iid for iid, _ in split_ids(cfg["split"]))
        # ADR-006, amendment 7: generations the provider failed in every attempt and launch.
        self.provider_errors = {
            (e["instance"], e["condition"], e["slot"]) for e in cfg.get("provider_errors", [])
        }

    def new_client(self, provider: str, concurrency: int = 1, **kw: Any):
        return super().new_client(
            provider, concurrency, filtered_log=self.out / "filtered.jsonl", **kw
        )

    # --- what was fixed before opening B --------------------------------------

    def check_decks(self) -> None:
        """The decks are the deck files' and the selections of the runs on split A."""
        origin = {
            "R": sorted(c.id for c in self.deck.values() if c.origin == "control"),
            "E6": sorted(c.id for c in self.deck.values() if c.origin == "paper"),
        }
        for name, expected in origin.items():
            if sorted(self.decks[name]) != expected:
                raise SystemExit(f"{name} must be every card of its origin: {expected}")
        selected = {
            "ML": json.loads(Path(self.cfg["map_metrics"]).read_text("utf-8"))["ml"]["cards"],
            "MG": json.loads(Path(self.cfg["g1_metrics"]).read_text("utf-8"))["mg"]["cards"],
        }
        for name, expected in selected.items():
            if sorted(self.decks[name]) != sorted(expected):
                raise SystemExit(f"{name} differs from the selection on split A: {expected}")

    def expected_draws(self) -> dict[str, dict[str, list[str]]]:
        return draw_cards(self.ids, self.decks)

    def check_draws(self) -> dict[str, dict[str, list[str]]]:
        draws = self.expected_draws()
        stored = json.loads(Path(self.cfg["draws"]).read_text("utf-8"))
        if stored != draws:
            raise SystemExit(f"{self.cfg['draws']} differs from the seeded draw")
        return draws

    def expected_wonder(self) -> list[list[Any]]:
        return draw_wonder_positions(self.ids, self.cfg["wonder"]["ideas_per_condition"])

    def check_wonder(self) -> list[tuple[str, int]]:
        positions = self.expected_wonder()
        stored = json.loads(Path(self.cfg["wonder"]["positions"]).read_text("utf-8"))
        if stored != positions:
            raise SystemExit(f"{self.cfg['wonder']['positions']} differs from the seeded draw")
        return [(iid, slot) for iid, slot in positions]

    def write_draws(self) -> Path:
        path = Path(self.cfg["draws"])
        text = json.dumps(self.expected_draws(), indent=1, sort_keys=True) + "\n"
        path.write_text(text, "utf-8", newline="\n")
        positions = Path(self.cfg["wonder"]["positions"])
        positions.write_text(json.dumps(self.expected_wonder()) + "\n", "utf-8", newline="\n")
        return path

    def seen_in_earlier_runs(self) -> list[str]:
        """Split B instances named in the records of any earlier run: there should be none."""
        seen: set[str] = set()
        for path in Path("runs").glob("*/*.jsonl"):
            if path.parent.name == self.run_id:
                continue
            with path.open(encoding="utf-8") as f:
                text = f.read()
            seen |= {iid for iid in self.ids if f'"{iid}"' in text}
        return sorted(seen)

    def instances(self) -> list[Instance]:
        """The only place that opens split B: it raises unless OBLIQUE_UNSEAL_B=1."""
        return sorted(load_instances(SPLIT), key=lambda x: x.id)

    # --- generation -----------------------------------------------------------

    async def generate_call(
        self, client, x, condition, k, sample_idx, previous=(), *, card=None, params=None
    ):
        """A generation listed as a provider error is not asked again (ADR-006, amendment 7)."""
        slot = sample_idx - SEQUENTIAL_SAMPLE_BASE
        if k != 1 or (x.id, condition, slot) not in self.provider_errors:
            return await super().generate_call(
                client, x, condition, k, sample_idx, previous, card=card, params=params
            )
        ideation = self.prompt("ideation")
        prompt_hash = f"{ideation.hash}+{self.prompt('inject').hash}" if card else ideation.hash
        self.progress["generations"] += 1
        return records_from_set(
            provider_error_set(1),
            cards=[card.id if card else None],
            run_id=self.run_id,
            instance=x.id,
            condition=condition,
            model=self.cfg["generator"]["model"],
            prompt_hash=prompt_hash,
            seed=None,
            call_key="provider_error",
            usage=Usage(),
        )

    async def generate(self, instances: list[Instance]) -> None:
        gen = self.cfg["generator"]
        # The timeout is not part of a cache key: it changes how long a call may take, not what
        # it returns.
        timeout = {"timeout": gen["timeout_seconds"]} if "timeout_seconds" in gen else {}
        client = self.new_client(gen["provider"], gen["concurrency"], **timeout)
        draws = self.check_draws()

        async def reference(x: Instance):
            return (x.id, "C1"), await self.generate_call(client, x, "C1", K, 0)

        async def sequence(x: Instance, condition: str):
            cards = None
            if condition in self.decks:
                cards = [self.deck[c] for c in draws[condition][x.id]]
            records = await self.generate_sequence(
                client, x, condition, cards=cards, params=self.no_card.get(condition)
            )
            return (x.id, condition), records

        jobs = [reference(x) for x in instances]
        jobs += [sequence(x, c) for x in instances for c in self.conditions if c != "C1"]
        _, results = await asyncio.gather(
            self.generate_anchors(client, instances), asyncio.gather(*jobs)
        )
        self.slots = dict(results)
        write_jsonl(
            self.out / "generations.jsonl",
            [r for key in sorted(self.slots) for r in self.slots[key]],
        )
        self.notes["generator_client"] = self.client_stats(client)

    # --- judges ---------------------------------------------------------------

    async def quality_wonder(self, client, x: Instance, cond: str, slot: int, idea) -> None:
        """Wonder against the anchor, both orders: agreement only (ADR-010, amendment 3)."""
        key = (x.id, cond, slot)
        anchor = self.anchors[x.id]
        if anchor is None:
            self.q["wonder"][key] = None
            return
        prompt = self.prompt("quality")
        mode = self.cfg["judges"]["wonder"]
        ids = {"instance": x.id, "condition": cond, "slot": slot}
        winners = []
        for idea_first in (True, False):
            messages = quality_messages(
                idea, render_card(anchor), idea_first=idea_first, prompt=prompt
            )
            winners.append(
                await self.judge_call(
                    client,
                    "quality-wonder",
                    prompt,
                    mode,
                    messages,
                    parse_winner,
                    ids | {"idea_first": idea_first},
                )
            )
        if None in winners:
            self.q["wonder"][key] = None
        else:
            self.q["wonder"][key] = q_from_orders(winners[0], winners[1])
            self.verdicts["wonder"][key] = (winners[0], winners[1])

    def wonder_jobs(self, client, instances: list[Instance]) -> list[Any]:
        by_id = {x.id: x for x in instances}
        jobs = []
        for iid, slot in self.check_wonder():
            for cond in self.conditions:
                record = next((r for r in self.slots[(iid, cond)] if r.slot == slot), None)
                if record is not None and record.valid:
                    jobs.append(self.quality_wonder(client, by_id[iid], cond, slot, record.idea))
        return jobs

    async def judge(self, instances: list[Instance]) -> None:
        client = self.judge_client("spark")
        wonder = self.judge_client("wonder")
        jobs = self.wonder_jobs(wonder, instances)
        for x in instances:
            for cond in self.conditions:
                valid = [(r.slot, r.idea) for r in self.slots[(x.id, cond)] if r.valid]
                for slot, idea in valid:
                    jobs.append(self.quality(client, "anchor", x, cond, slot, idea))
                    jobs.append(self.grounding_one(client, x, cond, slot, idea))
                    if self.cfg["q_gold"]:
                        jobs.append(self.quality(client, "spark", x, cond, slot, idea))
                for (i, a), (j, b) in itertools.combinations(valid, 2):
                    jobs.append(self.distance_pair(client, x, cond, i, j, a, b))
        try:
            await asyncio.gather(*jobs)
        finally:
            self.notes["judge_client"] = self.client_stats(client)
            self.notes["wonder_client"] = self.client_stats(wonder)

    # --- report ---------------------------------------------------------------

    def wonder_report(self, n_boot: int) -> dict[str, Any]:
        """Agreement with Spark on the drawn ideas, and ML - C1s on those ideas per judge. An
        invalid idea has q = 0 for both judges and enters only the difference, as in CU."""
        positions = self.check_wonder()
        judged = [
            (iid, cond, slot)
            for iid, slot in positions
            for cond in self.conditions
            if self.q["wonder"].get((iid, cond, slot)) is not None
            and self.q["anchor"].get((iid, cond, slot)) is not None
        ]
        ordered = [
            k for k in judged if k in self.verdicts["wonder"] and k in self.verdicts["anchor"]
        ]

        def value(judge: str, iid: str, cond: str, slot: int) -> float | None:
            record = next((r for r in self.slots[(iid, cond)] if r.slot == slot), None)
            if record is None or not record.valid:
                return 0.0
            return self.q[judge].get((iid, cond, slot))

        def difference(judge: str) -> dict[str, Any] | None:
            by_instance: dict[str, list[float]] = {}
            for iid, slot in positions:
                ml, base = (value(judge, iid, c, slot) for c in (PRIMARY, BASELINE))
                if ml is not None and base is not None:
                    by_instance.setdefault(iid, []).append(ml - base)
            means = [sum(v) / len(v) for v in by_instance.values()]
            return ci_dict(bootstrap_mean_ci(means, n_boot=n_boot)) if means else None

        return {
            "ideas_per_condition": len(positions),
            "ideas_judged": len(judged),
            "agreement_q": agreement(
                [self.q["anchor"][k] for k in judged], [self.q["wonder"][k] for k in judged]
            )
            if judged
            else None,
            "agreement_per_order": agreement(
                [v for k in ordered for v in self.verdicts["anchor"][k]],
                [v for k in ordered for v in self.verdicts["wonder"][k]],
            )
            if ordered
            else None,
            "delta_q_ml_c1s_wonder": difference("wonder"),
            "delta_q_ml_c1s_spark_same_ideas": difference("anchor"),
        }

    def set_metrics(self, key, provider_error_as_invalid: bool = False) -> dict[str, Any]:
        """A set with a provider error is incomplete, so its instance leaves the comparisons of that
        condition; as a sensitivity analysis, the idea counts as invalid, q = 0 (amendment 7)."""
        out = super().set_metrics(key)
        if any(r.invalid_reason == InvalidReason.provider_error for r in self.slots[key]):
            out["provider_error"] = True
            if not provider_error_as_invalid:
                out["complete"] = False
        return out

    def regression_rows(self) -> list[dict[str, Any]]:
        """One row per judgment of this run. An invalid or filtered idea is two losses at equal
        length; an idea missing one order is left out (ADR-003), and so is a provider error, which
        has no judgment (ADR-006, amendment 7)."""
        rows: list[dict[str, Any]] = []
        for (iid, cond), records in sorted(self.slots.items()):
            anchor = self.anchors.get(iid)
            if anchor is None:
                continue
            for r in records:
                verdict = self.verdicts["anchor"].get((iid, cond, r.slot))
                if r.invalid_reason == InvalidReason.provider_error:
                    continue
                if not r.valid:
                    wins, delta = (False, False), 0
                elif verdict is not None:
                    wins = (verdict[0] == "A", verdict[1] == "B")
                    delta = judged_words(r.idea) - judged_words(anchor)
                else:
                    continue
                rows += [
                    {"instance": iid, "condition": cond, "win": w, "delta_words": delta}
                    for w in wins
                ]
        return rows

    def paired(
        self, by: dict[str, dict[str, Any]], metric: str, a: str, b: str, n_boot: int, bca=False
    ) -> dict[str, Any] | None:
        x = {i: r[metric] for i, r in by[a].items() if r.get(metric) is not None}
        y = {i: r[metric] for i, r in by[b].items() if r.get(metric) is not None}
        common = sorted(set(x) & set(y))
        if not common:
            return None
        differences = paired_differences({i: x[i] for i in common}, {i: y[i] for i in common})
        cell = ci_dict(bootstrap_mean_ci(differences, n_boot=n_boot))
        if bca:
            check = bca_mean_ci(differences, n_boot=n_boot)
            cell |= {"bca_lo": check.lo, "bca_hi": check.hi}
        return cell

    def report(self, instances: list[Instance]) -> dict[str, Any]:
        n_h1, n_other = self.cfg["n_boot_h1"], self.cfg["n_boot_other"]
        delta = self.cfg["delta"]
        sets = [self.set_metrics(k) for k in sorted(self.slots)]
        complete = [r for r in sets if r["complete"]]
        by = {c: {r["instance"]: r for r in complete if r["condition"] == c}
              for c in self.conditions}  # fmt: skip
        rows = self.regression_rows()

        h1_q = controlled_win_rates(
            rows, self.conditions, comparisons=[(PRIMARY, BASELINE)], n_boot=n_h1, bca=True
        )
        others = [(c, BASELINE) for c in self.conditions if c not in (BASELINE, PRIMARY)]
        other_q = controlled_win_rates(rows, self.conditions, comparisons=others, n_boot=n_other)

        delta_cu = self.paired(by, "cu", PRIMARY, BASELINE, n_h1, bca=True)
        delta_q = h1_q["differences"][f"{PRIMARY}-{BASELINE}"]
        guard = self.paired(by, "grounding", PRIMARY, BASELINE, n_h1)

        def compare(a: str, b: str, metric: str = "cu") -> dict[str, Any] | None:
            return self.paired(by, metric, a, b, n_other)

        as_invalid = [
            self.set_metrics(k, provider_error_as_invalid=True) for k in sorted(self.slots)
        ]
        by_as_invalid = {
            c: {r["instance"]: r for r in as_invalid if r["complete"] and r["condition"] == c}
            for c in self.conditions
        }
        affected = any(
            r.invalid_reason == InvalidReason.provider_error
            for records in self.slots.values()
            for r in records
        )

        table = {}
        for cond in self.conditions:
            records = [r for (_, c), rs in self.slots.items() if c == cond for r in rs]
            cell: dict[str, Any] = {
                "sets_complete": len(by[cond]),
                "ideas": len(records),
                "invalid": sum(
                    not r.valid
                    and r.invalid_reason
                    not in (InvalidReason.content_filtered, InvalidReason.provider_error)
                    for r in records
                ),
                "provider_error": sum(
                    r.invalid_reason == InvalidReason.provider_error for r in records
                ),
                "filtered": sum(
                    r.invalid_reason == InvalidReason.content_filtered for r in records
                ),
                "q_controlled": other_q["rates"][cond],
            }
            for metric in ("cu", "q", "q_gold", "d", "d_embed", "grounding", "cu_strict"):
                values = [r[metric] for r in by[cond].values() if r.get(metric) is not None]
                cell[metric] = (
                    ci_dict(bootstrap_mean_ci(values, n_boot=n_other)) if values else None
                )
            table[cond] = cell

        pairs = [(d, self.d_emb[k]) for k, d in self.d_llm.items() if d is not None]
        return {
            "run_id": self.run_id,
            "instances": len(instances),
            "spent_usd": round(self.spent_usd, 4),
            "h1": {
                "verdict": h1_verdict(delta_cu, delta_q, delta),
                "delta_cu_ml_c1s": delta_cu,
                "delta_q_controlled_ml_c1s": delta_q,
                "delta": delta,
                "n_boot": n_h1,
            },
            "grounding_guard": {
                "delta_grounding_ml_c1s": guard,
                "threshold": self.cfg["grounding_guard"],
                "mean_above": guard["mean"] > self.cfg["grounding_guard"],
                "lower_bound_above": guard["lo"] > self.cfg["grounding_guard"],
            },
            "exploratory": {
                "n_boot": n_other,
                "h3_cu_ml_ts": compare("ML", "Ts"),
                "h3_cu_ml_r": compare("ML", "R"),
                "h4_cu_mg_ml": compare("MG", "ML"),
                "h6_cu_e6_c1s": compare("E6", "C1s"),
                "h6_q_controlled_e6_c1s": other_q["differences"]["E6-C1s"],
                "h7_cu_e6_ml": compare("E6", "ML"),
                "delta_grounding_vs_c1s": {
                    c: compare(c, "C1s", "grounding") for c in ("E6", "ML", "MG", "R")
                },
                "reference_cu_vs_c1": {c: compare(c, "C1") for c in ("C1s", "ML")},
            },
            "sensitivity": {
                "delta_q_raw_ml_c1s": compare("ML", "C1s", "q"),
                "delta_cu_strict_ml_c1s": compare("ML", "C1s", "cu_strict"),
                "delta_cu_embed_ml_c1s": compare("ML", "C1s", "cu_embed"),
                "wonder": self.wonder_report(n_other),
                "provider_error_as_invalid": {
                    "h3_cu_ml_ts": self.paired(by_as_invalid, "cu", "ML", "Ts", n_other),
                    "h3_cu_ml_r": self.paired(by_as_invalid, "cu", "ML", "R", n_other),
                }
                if affected
                else None,
            },
            "provider_errors": [
                {"instance": iid, "condition": cond, "slot": r.slot}
                for (iid, cond), records in sorted(self.slots.items())
                for r in records
                if r.invalid_reason == InvalidReason.provider_error
            ],
            "conditions": table,
            "length": {
                "log_odds_per_10_words": h1_q["log_odds_per_unit"],
                "judgments_in_regression": h1_q["judgments"],
            },
            "spearman_d_llm_vs_embed": {
                "n": len(pairs),
                "rho": spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None,
            },
            "content_filtered": self.content_filtered,
            "incomplete_sets": len(sets) - len(complete),
            "sets": sets,
            "notes": self.notes,
        }

    # --- the gate -------------------------------------------------------------

    def plan(self) -> dict[str, Any]:
        """Calls, cost and time from the instance IDs alone. Split B stays sealed."""
        n = len(self.ids)
        rates = self.cfg["estimate"]
        sequential = len(self.conditions) - 1
        ideas = n * K * len(self.conditions)
        quality = 2 * ideas
        counts = {
            "anchor": n,
            "c1": n,
            "sequential": n * K * sequential,
            "quality": quality,
            "quality_gold": quality if self.cfg["q_gold"] else 0,
            "quality_wonder": 2 * self.cfg["wonder"]["ideas_per_condition"] * len(self.conditions),
            "grounding": ideas,
            "distance": n * len(self.conditions) * K * (K - 1) // 2,
        }
        blocks = {
            name: {"calls": c, "usd": round(c * rates[name]["usd"], 3)}
            for name, c in counts.items()
        }
        generation = counts["anchor"] + counts["c1"] + counts["sequential"]
        judges = sum(counts.values()) - generation
        fit = bootstrap_seconds(quality, len(self.conditions))
        fits = self.cfg["n_boot_h1"] + self.cfg["n_boot_other"] + n
        return {
            "run_id": self.run_id,
            "split": SPLIT,
            "instances": n,
            "split_b_opened": False,
            "b_instances_in_earlier_runs": self.seen_in_earlier_runs(),
            "draws_file": self.cfg["draws"],
            "draws_sha256": hashlib.sha256(Path(self.cfg["draws"]).read_bytes()).hexdigest()[:16],
            "wonder_positions_sha256": hashlib.sha256(
                Path(self.cfg["wonder"]["positions"]).read_bytes()
            ).hexdigest()[:16],
            "conditions": {
                c: {
                    "kind": "one call" if c == "C1" else "sequential",
                    "params": self.no_card.get(c, {}),
                    "deck": self.decks.get(c),
                }
                for c in self.conditions
            },
            "regression": {
                "judgments": quality,
                "conditions": len(self.conditions),
                "data": "judgments of split B only",
                "refit_on_every_resample": True,
                "n_boot_h1": self.cfg["n_boot_h1"],
                "n_boot_other": self.cfg["n_boot_other"],
                "bca_check_on_h1": True,
            },
            "blocks": blocks,
            "calls_total": sum(counts.values()),
            "usd": round(sum(b["usd"] for b in blocks.values()), 2),
            "approved_usd": self.cfg["budget"]["approved_usd"],
            "generation_minutes": round(
                generation
                * rates["sequential"]["seconds"]
                / self.cfg["generator"]["concurrency"]
                / 60,
                1,
            ),  # fmt: skip
            "judge_minutes": round(judges / self.cfg["estimate_judge_calls_per_minute"], 1),
            "bootstrap_minutes": round(fit * fits / 60, 1),
        }

    # --- run ------------------------------------------------------------------

    async def run(self) -> None:
        metrics = self.out / "metrics.json"
        if metrics.exists():
            raise SystemExit(f"{metrics} exists: split B runs once (ADR-010, amendment 2)")
        self.check_decks()
        self.check_draws()
        self.check_wonder()
        instances = self.instances()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), "utf-8")
        try:
            await self.check_models()
            await self.generate(instances)
            print("generation done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)
            await self.judge(instances)
            print("judges done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)
            self.embed()
            result = self.report(instances)
            (self.out / "summary.md").write_text(render_summary(result), "utf-8", newline="\n")
            metrics.write_text(
                json.dumps(result, indent=2, default=str, ensure_ascii=False), "utf-8"
            )
        finally:
            write_jsonl_dicts(self.out / "judgments.jsonl", self.judge_rows)
            write_jsonl_dicts(self.out / "calls.jsonl", self.calls)
            print(f"spent US${self.spent_usd:.4f}; {len(self.calls)} calls", flush=True)


def _ci(cell: dict[str, Any] | None) -> str:
    return f"{cell['mean']:.3f} [{cell['lo']:.3f}, {cell['hi']:.3f}]" if cell else "n/a"


def _ci_n(cell: dict[str, Any] | None) -> str:
    return f"{_ci(cell)} (n = {cell['n']})" if cell else "n/a"


def _agreement(cell: dict[str, Any] | None) -> str:
    return f"{cell['exact']:.1%}, kappa {cell['kappa']:.2f}" if cell else "n/a"


def render_summary(result: dict[str, Any]) -> str:
    h1, guard, ex = result["h1"], result["grounding_guard"], result["exploratory"]
    sens = result["sensitivity"]
    wonder = sens["wonder"]
    cu = h1["delta_cu_ml_c1s"]
    lines = [
        f"# {result['run_id']} · Confirmation on split B",
        "",
        f"{result['instances']} instances, US${result['spent_usd']:.2f}, one run.",
        "",
        "## H1",
        "",
        f"**Verdict: {h1['verdict']}** (ADR-003).",
        "",
        f"- CU(ML) - CU(C1s): {_ci(cu)}; BCa check [{cu['bca_lo']:.3f}, {cu['bca_hi']:.3f}].",
        f"- Controlled q, ML - C1s: {_ci(h1['delta_q_controlled_ml_c1s'])}; BCa check "
        f"[{h1['delta_q_controlled_ml_c1s']['bca_lo']:.3f}, "
        f"{h1['delta_q_controlled_ml_c1s']['bca_hi']:.3f}]; margin -{h1['delta']}.",
        "",
        "## Grounding guard",
        "",
        f"Grounding, ML - C1s: {_ci(guard['delta_grounding_ml_c1s'])}; threshold "
        f"{guard['threshold']}. Reported; it does not decide H1.",
        "",
        "## Exploratory (post-map hypotheses are marked in ADR-010)",
        "",
        f"- H3, CU(ML) - CU(Ts): {_ci_n(ex['h3_cu_ml_ts'])}",
        f"- H3, CU(ML) - CU(R): {_ci_n(ex['h3_cu_ml_r'])}",
        f"- H4, CU(MG) - CU(ML): {_ci_n(ex['h4_cu_mg_ml'])}",
        f"- H6, CU(E6) - CU(C1s): {_ci_n(ex['h6_cu_e6_c1s'])}",
        f"- H6, controlled q, E6 - C1s: {_ci(ex['h6_q_controlled_e6_c1s'])}",
        f"- H7, CU(E6) - CU(ML): {_ci_n(ex['h7_cu_e6_ml'])}",
        "",
        "## Sensitivity (does not decide H1)",
        "",
        f"- Raw q, ML - C1s: {_ci(sens['delta_q_raw_ml_c1s'])}",
        f"- CU under the strict rule, ML - C1s: {_ci(sens['delta_cu_strict_ml_c1s'])}",
        f"- CU with embedding distance, ML - C1s: {_ci(sens['delta_cu_embed_ml_c1s'])}",
        f"- Wonder on {wonder['ideas_judged']} ideas: agreement with Spark "
        f"{_agreement(wonder['agreement_q'])} per idea, "
        f"{_agreement(wonder['agreement_per_order'])} per order.",
        f"- q, ML - C1s, on those ideas: Wonder {_ci(wonder['delta_q_ml_c1s_wonder'])}; "
        f"Spark {_ci(wonder['delta_q_ml_c1s_spark_same_ideas'])}.",
        *(
            [
                f"- H3 with the provider error as q = 0 (ADR-006, amendment 7): CU(ML) - CU(Ts) "
                f"{_ci_n(sens['provider_error_as_invalid']['h3_cu_ml_ts'])}."
            ]
            if sens["provider_error_as_invalid"]
            else []
        ),
        "",
        "## Conditions",
        "",
        "| Condition | Sets | CU | q controlled | q raw | q gold | d | Grounding | Invalid "
        "| Filtered | Provider error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cond, c in result["conditions"].items():
        lines.append(
            f"| {cond} | {c['sets_complete']} | {_ci(c['cu'])} | {_ci(c['q_controlled'])} "
            f"| {_ci(c['q'])} | {_ci(c['q_gold'])} | {_ci(c['d'])} "
            f"| {_ci(c['grounding'])} | {c['invalid']} | {c['filtered']} | {c['provider_error']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    flags = ("--plan", "--write-draws")
    if len(argv) not in (1, 2) or (len(argv) == 2 and argv[1] not in flags):
        raise SystemExit("usage: python -m oblique.confirm configs/R-*.yaml [--plan|--write-draws]")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    run = ConfirmB(cfg)
    if argv[1:] == ["--write-draws"]:
        run.check_decks()
        print(run.write_draws())
        return
    if argv[1:] == ["--plan"]:
        run.check_decks()
        run.check_draws()
        run.check_wonder()
        print(json.dumps(run.plan(), indent=2))
        return
    load_env()
    asyncio.run(run.run())


if __name__ == "__main__":
    main(sys.argv[1:])
