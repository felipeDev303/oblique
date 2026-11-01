"""The map of the deck on split A: one idea per card and instance (mode 1).

    uv run python -m oblique.map_sweep configs/R-*.yaml --plan     # counts and cost, no API calls
    uv run --group embed python -m oblique.map_sweep configs/R-*.yaml

For each instance: the anchor and S0 (the set of C1s, ADR-009), then one K = 1 idea per
condition: every configured card, C2 (no card) and C2-T (no card, high temperature). Each idea is
judged against the anchor in both orders, for grounding, and for distance to each idea of S0.

Resuming: every successful call is in the response cache, so rerunning the same config pays only
for what is missing, and `budget.ledger` keeps the spend across launches.

Metrics per condition, with percentile CIs over instances:
- q controlled for length (primary, ADR-009): joint logistic regression at equal length;
- q raw: mean of q in {0, 0.5, 1};
- d: mean over instances of min over S0 of the LLM distance; embedding d is secondary;
- m: mean of q raw * d, the marginal value that feeds CU;
- grounding, and the rates of invalid and content-filtered ideas.
An invalid idea has q = 0, m = 0 and no d or grounding (ADR-006); in the regression it counts as
a loss at equal length. A missing judgment leaves that value empty, and the instance out of the
paired comparisons that need it (ADR-006, amendment 6).

Bands (ADR-003), from paired CIs against C2: see `classify_band`. The ML deck: see `select_ml`.
CIs use 2,000 resamples, refitting the regression each time (ADR-003, `n_boot` in the config).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from oblique.baseline import ANCHOR_SAMPLE_IDX, SEQUENTIAL_SAMPLE_BASE, K, ci_dict
from oblique.bench import Instance, load_instances
from oblique.cache import make_key
from oblique.deck import Card, load_deck
from oblique.deck.controls import DECK_DIR, pick_controls
from oblique.generate.labels import LabelStatus
from oblique.generate.records import write_jsonl
from oblique.generate.validate import InvalidReason, parse_set
from oblique.length_check import LengthCheck
from oblique.metrics import bootstrap_mean_ci, paired_delta_ci
from oblique.metrics.calibration import controlled_win_rates, pearson, spearman
from oblique.prompts import ideation_messages, judged_words
from oblique.smoke import load_env, write_jsonl_dicts

CONDITION_SAMPLE_IDX = 600  # reserved: a card or C2 never shares a draw with an anchor or S0
S0 = "C1s"
REFERENCE = "C2"
N_CONTROLS_IN_MAP = 10
MIN_LIMINAL = 3  # ADR-003: fewer liminal OB cards than this and ML falls back
FALLBACK_SIZE = 6


def classify_band(
    dq_lo: float, dq_hi: float, dd_lo: float, dd_hi: float, delta: float
) -> dict[str, Any]:
    """ADR-003 bands from the paired CIs against C2 of q controlled for length (dq) and d (dd).

    destructive: upper bound of dq < -delta, and it takes priority. Otherwise, from dd alone:
    liminal if its lower bound > 0, convergent if its upper bound < 0, inert if it contains 0.
    A liminal card is "confirmed" if also the lower bound of dq > -delta.
    """
    destructive = dq_hi < -delta
    if destructive:
        band = "destructive"
    elif dd_lo > 0:
        band = "liminal"
    elif dd_hi < 0:
        band = "convergent"
    else:
        band = "inert"
    return {
        "band": band,
        "confirmed": band == "liminal" and dq_lo > -delta,
        "destructive": destructive,
    }


def select_ml(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The ML deck (ADR-003): every liminal OB card; with fewer than 3, the 6 non-destructive
    OB cards with the highest m, marked as fallback. E6 and CTRL-R cards never enter."""
    seed = [r for r in rows if r["origin"] == "seed"]
    liminal = sorted(r["condition"] for r in seed if r["band"] == "liminal")
    if len(liminal) >= MIN_LIMINAL:
        return {"cards": liminal, "fallback": False, "liminal_ob": liminal}
    candidates = [r for r in seed if r["band"] != "destructive" and r["m"] is not None]
    candidates.sort(key=lambda r: (-r["m"]["mean"], r["condition"]))
    return {
        "cards": [r["condition"] for r in candidates[:FALLBACK_SIZE]],
        "fallback": True,
        "liminal_ob": liminal,
    }


class MapSweep(LengthCheck):
    def __init__(self, cfg: dict[str, Any], **kw: Any):
        super().__init__(cfg, **kw)
        deck = load_deck(cfg.get("deck", DECK_DIR))
        chosen = [card for group in cfg["cards"].values() for card in group]
        missing = sorted(set(chosen) - set(deck))
        if missing:
            raise SystemExit(f"cards not in the deck: {missing}")
        self.deck = deck
        self.cards: dict[str, Card] = {c: deck[c] for c in chosen}
        self.no_card: dict[str, dict[str, Any]] = cfg["no_card"]
        self.conditions = (*chosen, *self.no_card)
        self.calls_seen = 0
        self.started = time.perf_counter()

    def check_cards(self) -> None:
        """The config lists every seed and paper card and the seeded draw of controls."""
        by_origin = {o: sorted(c.id for c in self.deck.values() if c.origin == o) for o in
                     ("seed", "paper", "control")}  # fmt: skip
        cards = self.cfg["cards"]
        if (
            sorted(cards["seed"]) != by_origin["seed"]
            or sorted(cards["paper"]) != by_origin["paper"]
        ):
            raise SystemExit("the config must list every seed and paper card")
        drawn = pick_controls(by_origin["control"], N_CONTROLS_IN_MAP)
        if sorted(cards["control"]) != drawn:
            raise SystemExit(f"config controls differ from the seeded draw: {drawn}")

    def instances(self) -> list[Instance]:
        return sorted(load_instances(self.cfg["split"]), key=lambda x: x.id)

    def comparisons(self) -> list[tuple[str, str]]:
        return [(c, REFERENCE) for c in self.conditions if c != REFERENCE]

    async def call(self, label, mode, messages, params=None, sample_idx=0, **kw):
        c = await super().call(label, mode, messages, params, sample_idx, **kw)
        self.calls_seen += 1
        if self.calls_seen % self.cfg.get("progress_every", 500) == 0:
            minutes = (time.perf_counter() - self.started) / 60
            print(
                f"{self.calls_seen} calls, US${self.spent_usd:.3f}, {minutes:.1f} min, "
                f"filtered {len(self.content_filtered)}",
                flush=True,
            )
        return c

    # --- generation -----------------------------------------------------------

    async def generate_condition(self, client, x: Instance, condition: str):
        return await self.generate_call(
            client,
            x,
            condition,
            1,
            CONDITION_SAMPLE_IDX,
            card=self.cards.get(condition),
            params=self.no_card.get(condition),
        )

    async def generate(self, instances: list[Instance]) -> None:
        gen = self.cfg["generator"]
        client = self.new_client(gen["provider"], gen["concurrency"])

        async def set_of(x: Instance):
            return (x.id, S0), await self.generate_sequence(client, x, S0)

        async def one(x: Instance, condition: str):
            return (x.id, condition), await self.generate_condition(client, x, condition)

        jobs = [set_of(x) for x in instances]
        jobs += [one(x, c) for x in instances for c in self.conditions]
        anchors, results = await asyncio.gather(
            self.generate_anchors(client, instances), asyncio.gather(*jobs)
        )
        del anchors
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
            s0 = [(r.slot, r.idea) for r in self.slots[(x.id, S0)] if r.valid]
            for cond in self.conditions:
                r = self.slots[(x.id, cond)][0]
                if not r.valid:
                    continue
                jobs.append(self.quality(client, "anchor", x, cond, r.slot, r.idea))
                jobs.append(self.grounding_one(client, x, cond, r.slot, r.idea))
                for s_slot, s_idea in s0:
                    jobs.append(self.distance_pair(client, x, cond, r.slot, s_slot, r.idea, s_idea))
        try:
            await asyncio.gather(*jobs)
        finally:
            self.notes["judge_client"] = self.client_stats(client)

    # --- report ---------------------------------------------------------------

    def embeddings(self) -> dict[tuple[str, str, int, int], float]:
        """Embedding distance for every idea-S0 pair the LLM judge saw."""
        from oblique.judges.embed import MODEL, REVISION, Embedder, pair_distance

        keys = [
            (iid, c, r.slot) for (iid, c), rs in sorted(self.slots.items()) for r in rs if r.valid
        ]
        ideas = [self.slots[(iid, c)][slot if c == S0 else 0].idea for iid, c, slot in keys]
        vectors = dict(zip(keys, (self.embedder or Embedder()).embed(ideas), strict=True))
        self.notes["embeddings"] = {"model": MODEL, "revision": REVISION, "ideas": len(ideas)}
        return {
            (iid, c, i, j): pair_distance(vectors[(iid, c, i)], vectors[(iid, S0, j)])
            for iid, c, i, j in self.d_llm
        }

    def instance_values(self, x: Instance, cond: str, d_emb) -> dict[str, Any]:
        """One condition on one instance. None marks a missing value (never an invalid idea)."""
        r = self.slots[(x.id, cond)][0]
        anchor = self.anchors.get(x.id)
        out: dict[str, Any] = {
            "invalid": not r.valid and r.invalid_reason != InvalidReason.content_filtered,
            "filtered": r.invalid_reason == InvalidReason.content_filtered,
        }
        if not r.valid:
            has_anchor = anchor is not None
            return out | {
                "q": 0.0 if has_anchor else None,
                "m": 0.0 if has_anchor else None,
                "d": None,
                "d_embed": None,
                "grounding": None,
                "words": None,
                "delta_words": None,
                "verdict": None,
            }
        s0 = [s.slot for s in self.slots[(x.id, S0)] if s.valid]
        pairs = [self.d_llm.get((x.id, cond, r.slot, s)) for s in s0]
        d = min(pairs) if pairs and all(v is not None for v in pairs) else None
        q = self.q["anchor"].get((x.id, cond, r.slot))
        embed = [d_emb[(x.id, cond, r.slot, s)] for s in s0 if (x.id, cond, r.slot, s) in d_emb]
        return out | {
            "q": q,
            "m": q * d if q is not None and d is not None else None,
            "d": d,
            "d_embed": min(embed) if embed else None,
            "grounding": self.ground.get((x.id, cond, r.slot)),
            "words": judged_words(r.idea),
            "delta_words": judged_words(r.idea) - judged_words(anchor) if anchor else None,
            "verdict": self.verdicts["anchor"].get((x.id, cond, r.slot)),
        }

    def regression_rows(self, values: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """One row per judgment. An invalid idea is two losses at equal length (q = 0)."""
        rows = []
        for cond, by_instance in values.items():
            for iid, v in by_instance.items():
                if v["verdict"] is not None and v["delta_words"] is not None:
                    idea_first, anchor_first = v["verdict"]
                    wins = (idea_first == "A", anchor_first == "B")
                    delta = v["delta_words"]
                elif (v["invalid"] or v["filtered"]) and v["q"] == 0.0:
                    wins, delta = (False, False), 0
                else:
                    continue
                rows += [
                    {"instance": iid, "condition": cond, "win": w, "delta_words": delta}
                    for w in wins
                ]
        return rows

    def label_rates(self, cond: str) -> dict[str, float]:
        records = [r for (_, c), rs in self.slots.items() if c == cond for r in rs if r.valid]
        off = (LabelStatus.off_list, LabelStatus.missing)
        n = len(records) or 1
        return {
            "opportunity_off_or_missing": sum(r.opportunity_label_status in off for r in records)
            / n,
            "method_off_or_missing": sum(r.method_label_status in off for r in records) / n,
        }

    def report(self, instances: list[Instance]) -> dict[str, Any]:
        d_emb = self.embeddings()
        values = {
            cond: {x.id: self.instance_values(x, cond, d_emb) for x in instances}
            for cond in self.conditions
        }

        n_boot = self.cfg["n_boot"]

        def column(cond: str, name: str) -> dict[str, float]:
            return {i: v[name] for i, v in values[cond].items() if v[name] is not None}

        def ci_or_none(data: list[float]) -> dict[str, float | int] | None:
            return ci_dict(bootstrap_mean_ci(data, n_boot=n_boot)) if data else None

        controlled = controlled_win_rates(
            self.regression_rows(values),
            self.conditions,
            comparisons=self.comparisons(),
            n_boot=n_boot,
        )
        delta = self.cfg["delta"]

        def paired(cond: str, name: str):
            a, b = column(cond, name), column(REFERENCE, name)
            common = set(a) & set(b)
            if cond == REFERENCE or not common:
                return None
            return ci_dict(
                paired_delta_ci({i: a[i] for i in common}, {i: b[i] for i in common}, n_boot=n_boot)
            )

        rows = []
        for cond in self.conditions:
            card = self.cards.get(cond)
            q_controlled = controlled["rates"][cond]
            dq = controlled["differences"].get(f"{cond}-{REFERENCE}")
            dd = paired(cond, "d")
            words = list(column(cond, "words").values())
            cell: dict[str, Any] = {
                "condition": cond,
                "text": card.text if card else None,
                "origin": card.origin if card else "no_card",
                "family": card.family.value if card and card.family else None,
                "q_controlled": q_controlled,
                "q_controlled_minus_c2": dq,
                "q_raw": ci_or_none(list(column(cond, "q").values())),
                "q_raw_minus_c2": paired(cond, "q"),
                "d": ci_or_none(list(column(cond, "d").values())),
                "d_minus_c2": dd,
                "d_embed": ci_or_none(list(column(cond, "d_embed").values())),
                "m": ci_or_none(list(column(cond, "m").values())),
                "m_minus_c2": paired(cond, "m"),
                "grounding": ci_or_none(list(column(cond, "grounding").values())),
                "grounding_minus_c2": paired(cond, "grounding"),
                "invalid_rate": ci_or_none([float(v["invalid"]) for v in values[cond].values()]),
                "filtered_rate": ci_or_none([float(v["filtered"]) for v in values[cond].values()]),
                "words_mean": sum(words) / len(words) if words else None,
                "delta_words_mean": (
                    sum(column(cond, "delta_words").values()) / len(column(cond, "delta_words"))
                    if column(cond, "delta_words")
                    else None
                ),
                **self.label_rates(cond),
            }
            if cond == REFERENCE:
                cell |= {"band": "reference"}
            elif dd is None:
                cell |= {"band": "unclassified"}
            else:
                cell |= classify_band(dq["lo"], dq["hi"], dd["lo"], dd["hi"], delta)
            rows.append(cell)

        pairs = [(v, d_emb[k]) for k, v in self.d_llm.items() if v is not None and k in d_emb]
        keys = [
            (i, c)
            for c in self.conditions
            for i, v in values[c].items()
            if v["q"] is not None and v["delta_words"] is not None
        ]
        qs = [values[c][i]["q"] for i, c in keys]
        return {
            "run_id": self.run_id,
            "spent_usd": round(self.spent_usd, 4),
            "instances": len(instances),
            "anchors_valid": sum(a is not None for a in self.anchors.values()),
            "s0_valid_ideas": sum(
                r.valid for (_, c), rs in self.slots.items() if c == S0 for r in rs
            ),
            "reference": {
                "q_controlled_c2": controlled["rates"][REFERENCE],
                "d_c2": ci_or_none(list(column(REFERENCE, "d").values())),
                "delta": delta,
                "n_boot": n_boot,
            },
            "length": {
                "log_odds_per_10_words": controlled["log_odds_per_unit"],
                "judgments_in_regression": controlled["judgments"],
                "pearson_delta_words_q": pearson(
                    [values[c][i]["delta_words"] for i, c in keys], qs
                ),
            },
            "spearman_d_llm_vs_embed": {
                "n": len(pairs),
                "rho": spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None,
            },
            "bands": {
                band: [r["condition"] for r in rows if r["band"] == band]
                for band in sorted({r["band"] for r in rows})
            },
            "liminal_confirmed": [r["condition"] for r in rows if r.get("confirmed")],
            "ml": select_ml(rows),
            "conditions": rows,
            "content_filtered": self.content_filtered,
            "missing_values": {
                name: sum(
                    v[name] is None and not v["invalid"] and not v["filtered"]
                    for by in values.values()
                    for v in by.values()
                )
                for name in ("q", "d", "grounding")
            },
            "notes": self.notes,
        }

    # --- gate -----------------------------------------------------------------

    def plan(self, instances: list[Instance]) -> dict[str, Any]:
        """What the run would call, how much of the generation is cached, cost and time."""
        gen = self.cfg["generator"]
        ideation = self.prompt("ideation")

        def is_cached(messages, params, sample_idx) -> bool:
            key = make_key(
                gen["model"],
                messages,
                {**gen["params"], **(params or {})},
                sample_idx,
                0,
                provider=gen["provider"],
            )
            return key in self.cache

        anchors = sum(
            is_cached(ideation_messages(x, 1, ideation), None, ANCHOR_SAMPLE_IDX) for x in instances
        )
        s0_cached = 0
        for x in instances:
            previous = []
            for slot in range(K):
                messages = ideation_messages(x, 1, ideation, previous)
                key = make_key(
                    gen["model"],
                    messages,
                    gen["params"],
                    SEQUENTIAL_SAMPLE_BASE + slot,
                    0,
                    provider=gen["provider"],
                )
                response = self.cache.get(key)
                if response is None:
                    break
                s0_cached += 1
                content = (response.get("choices") or [{}])[0].get("message", {}).get("content")
                parsed = parse_set(content, 1)
                previous = [*previous, *parsed.valid]
        n, conds = len(instances), len(self.conditions)
        conditions_cached = sum(
            is_cached(
                ideation_messages(
                    x, 1, ideation, card=self.cards[c].text if c in self.cards else None
                ),
                self.no_card.get(c),
                CONDITION_SAMPLE_IDX,
            )
            for x in instances
            for c in self.conditions
        )
        ideas = n * conds
        counts = {
            "anchor": (n, anchors),
            "s0": (n * K, s0_cached),
            "condition": (ideas, conditions_cached),
            "quality": (2 * ideas, 0),
            "grounding": (ideas, 0),
            "distance": (K * ideas, 0),
        }
        rates = self.cfg["estimate"]
        blocks = {
            name: {
                "calls": total,
                "cached": cached,
                "to_pay": total - cached,
                "usd": (total - cached) * rates[name]["usd"],
            }
            for name, (total, cached) in counts.items()
        }
        work = lambda names: sum(blocks[b]["to_pay"] * rates[b]["seconds"] for b in names)  # noqa: E731
        judges = ("quality", "grounding", "distance")
        spark = self.cfg["judges"]["spark"]
        return {
            "instances": n,
            "conditions": conds,
            "ideas": ideas,
            "blocks": blocks,
            "calls_total": sum(b["calls"] for b in blocks.values()),
            "calls_to_pay": sum(b["to_pay"] for b in blocks.values()),
            "usd": sum(b["usd"] for b in blocks.values()),
            "generation_minutes": work(("anchor", "s0", "condition")) / gen["concurrency"] / 60,
            "judge_minutes_at_start_limit": work(judges) / spark["concurrency"] / 60,
            "judge_minutes_at_ceiling": work(judges)
            / spark.get("max_concurrency", spark["concurrency"])
            / 60,
        }

    # --- run ------------------------------------------------------------------

    async def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), "utf-8")
        try:
            await self.check_models()
            self.check_cards()
            instances = self.instances()
            await self.generate(instances)
            print("generation done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)
            await self.judge(instances)
            print("judges done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)
            result = self.report(instances)
            (self.out / "metrics.json").write_text(
                json.dumps(result, indent=2, default=str, ensure_ascii=False), "utf-8"
            )
            (self.out / "summary.md").write_text(render_summary(result), "utf-8", newline="\n")
        finally:
            write_jsonl_dicts(self.out / "judgments.jsonl", self.judge_rows)
            write_jsonl_dicts(self.out / "calls.jsonl", self.calls)
            print(f"spent US${self.spent_usd:.4f}; {self.calls_seen} calls seen", flush=True)


def _ci(cell: dict[str, Any] | None) -> str:
    return f"{cell['mean']:.3f} [{cell['lo']:.3f}, {cell['hi']:.3f}]" if cell else "n/a"


def render_summary(result: dict[str, Any]) -> str:
    ref = result["reference"]
    lines = [
        f"# {result['run_id']} · Map of the deck on split A",
        "",
        f"{result['instances']} instances, {len(result['conditions'])} conditions, "
        f"US${result['spent_usd']:.2f}. Reference C2: controlled q = {_ci(ref['q_controlled_c2'])}, "
        f"d = {_ci(ref['d_c2'])}; delta = {ref['delta']}. "
        f"Spearman LLM vs embedding d = {result['spearman_d_llm_vs_embed']['rho']:.2f}. "
        f"Log-odds per 10 words: {_ci(result['length']['log_odds_per_10_words'])}.",
        "",
        f"**ML ({'fallback' if result['ml']['fallback'] else 'liminal OB cards'}):** "
        f"{', '.join(result['ml']['cards']) or 'none'}.",
        "",
        f"Values are means with 95% percentile CIs over instances ({ref['n_boot']} resamples), "
        "uncorrected across cards. Bands per ADR-003. Sorted by m.",
        "",
        "| Condition | Band | Δq controlled | q controlled | q raw | Δd | d | m | Grounding "
        "| Invalid | Filtered |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = sorted(result["conditions"], key=lambda r: -(r["m"]["mean"] if r["m"] else -1))
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['band']}{' (confirmed)' if r.get('confirmed') else ''} "
            f"| {_ci(r['q_controlled_minus_c2'])} | {_ci(r['q_controlled'])} | {_ci(r['q_raw'])} "
            f"| {_ci(r['d_minus_c2'])} | {_ci(r['d'])} | {_ci(r['m'])} | {_ci(r['grounding'])} "
            f"| {r['invalid_rate']['mean']:.3f} | {r['filtered_rate']['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if len(argv) not in (1, 2) or (len(argv) == 2 and argv[1] != "--plan"):
        raise SystemExit("usage: python -m oblique.map_sweep configs/R-*.yaml [--plan]")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    sweep = MapSweep(cfg)
    if len(argv) == 2:
        sweep.check_cards()
        print(json.dumps(sweep.plan(sweep.instances()), indent=2))
        return
    load_env()
    asyncio.run(sweep.run())


if __name__ == "__main__":
    main(sys.argv[1:])
