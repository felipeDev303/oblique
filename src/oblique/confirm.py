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

from oblique.baseline import Baseline, K, ci_dict
from oblique.bench import Instance, load_instances, split_ids
from oblique.deck import Card, load_deck
from oblique.deck.controls import DECK_DIR
from oblique.generate.records import write_jsonl
from oblique.generate.validate import InvalidReason
from oblique.metrics import paired_differences
from oblique.metrics.bootstrap import bca_mean_ci, bootstrap_mean_ci
from oblique.metrics.calibration import controlled_win_rates, fit_condition_logit, spearman
from oblique.prompts import judged_words
from oblique.seeds import seed_sequence
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

    def write_draws(self) -> Path:
        path = Path(self.cfg["draws"])
        text = json.dumps(self.expected_draws(), indent=1, sort_keys=True) + "\n"
        path.write_text(text, "utf-8", newline="\n")
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

    async def generate(self, instances: list[Instance]) -> None:
        gen = self.cfg["generator"]
        client = self.new_client(gen["provider"], gen["concurrency"])
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

    async def judge(self, instances: list[Instance]) -> None:
        client = self.judge_client("spark")
        jobs = []
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

    # --- report ---------------------------------------------------------------

    def regression_rows(self) -> list[dict[str, Any]]:
        """One row per judgment of this run. An invalid or filtered idea is two losses at equal
        length; an idea missing one order is left out (ADR-003)."""
        rows: list[dict[str, Any]] = []
        for (iid, cond), records in sorted(self.slots.items()):
            anchor = self.anchors.get(iid)
            if anchor is None:
                continue
            for r in records:
                verdict = self.verdicts["anchor"].get((iid, cond, r.slot))
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

        table = {}
        for cond in self.conditions:
            records = [r for (_, c), rs in self.slots.items() if c == cond for r in rs]
            cell: dict[str, Any] = {
                "sets_complete": len(by[cond]),
                "ideas": len(records),
                "invalid": sum(
                    not r.valid and r.invalid_reason != InvalidReason.content_filtered
                    for r in records
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
            },
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


def render_summary(result: dict[str, Any]) -> str:
    h1, guard, ex = result["h1"], result["grounding_guard"], result["exploratory"]
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
        f"- H3, CU(ML) - CU(Ts): {_ci(ex['h3_cu_ml_ts'])}",
        f"- H3, CU(ML) - CU(R): {_ci(ex['h3_cu_ml_r'])}",
        f"- H4, CU(MG) - CU(ML): {_ci(ex['h4_cu_mg_ml'])}",
        f"- H6, CU(E6) - CU(C1s): {_ci(ex['h6_cu_e6_c1s'])}",
        f"- H6, controlled q, E6 - C1s: {_ci(ex['h6_q_controlled_e6_c1s'])}",
        f"- H7, CU(E6) - CU(ML): {_ci(ex['h7_cu_e6_ml'])}",
        "",
        "## Conditions",
        "",
        "| Condition | CU | q controlled | q raw | d | Grounding | Invalid | Filtered |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cond, c in result["conditions"].items():
        lines.append(
            f"| {cond} | {_ci(c['cu'])} | {_ci(c['q_controlled'])} | {_ci(c['q'])} | {_ci(c['d'])} "
            f"| {_ci(c['grounding'])} | {c['invalid']} | {c['filtered']} |"
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
        print(json.dumps(run.plan(), indent=2))
        return
    load_env()
    asyncio.run(run.run())


if __name__ == "__main__":
    main(sys.argv[1:])
