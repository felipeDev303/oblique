"""Baseline C0 / C1 on split A with judge calibration (R-20261030-03).

    uv run --group embed python -m oblique.baseline configs/R-20261030-03.yaml

C0: three independent single-idea samples (K = 1, sample_idx 0-2) form a set.
C1: one call that asks for a set of K = 3. No card, no seed.

Validity follows ADR-006: one retry (attempt 1) only on `json_error`; invalid
ideas get q = 0 and are not judged. Judges (author, 2026-10-30):
- quality, Spark, both orders against the anchor idea (primary q, ADR-008), with
  `q_reference: anchor`; a run configured with `q_reference: gold` predates it;
- quality, Spark, both orders against the gold (secondary q);
- quality, Wonder, both orders against the gold (agreement only);
- distance, Spark, every pair of valid ideas in a set (primary d);
- grounding, Spark, every valid idea;
- distance from local embeddings, the same pairs (secondary d).
A judge reply that breaks its output contract is retried once (attempt 1);
if it fails again the value is missing, and a set missing a q or a d is left
out of CU and counted.

Outputs in runs/<run_id>/: generations.jsonl, judgments.jsonl, calls.jsonl,
metrics.json, summary.md.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from oblique.bench import Instance, load_instances, pick_baseline
from oblique.deck import Card
from oblique.generate.records import GenerationRecord, records_from_set, write_jsonl
from oblique.generate.schema import IdeaCard
from oblique.generate.validate import filtered_set, parse_set
from oblique.llm import ContentFilteredError, Usage
from oblique.metrics import bootstrap_mean_ci, creative_utility, paired_delta_ci
from oblique.metrics.calibration import agreement, pearson, position_consistency, spearman
from oblique.prompts import (
    DISTANCE_V1,
    GROUNDING_V1,
    PROMPTS,
    JudgeOutputError,
    Prompt,
    distance,
    distance_messages,
    grounding,
    grounding_messages,
    ideation_messages,
    judged_words,
    parse_grounding,
    parse_similarity,
    parse_winner,
    q_from_orders,
    quality_messages,
    render_card,
)
from oblique.smoke import MissingModels, Smoke, load_env, write_jsonl_dicts

K = 3
CONDITIONS = ("C0", "C1")
ANCHOR_SAMPLE_IDX = 500  # reserved (ADR-008): one anchor per instance, shared by all conditions
SEQUENTIAL_SAMPLE_BASE = 100  # the first idea of a sequence reads like C0's: never the same draw
SetKey = tuple[str, str]  # (instance, condition)


def ci_dict(ci) -> dict[str, float | int]:
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


class Baseline(Smoke):
    def __init__(self, cfg: dict[str, Any], *, embedder: Any = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.embedder = embedder  # tests only: stands in for the sentence-transformers model
        self.slots: dict[SetKey, list[GenerationRecord]] = {}
        self.judge_rows: list[dict[str, Any]] = []
        # "spark" and "wonder": q against the gold; "anchor": Spark, q against the anchor (ADR-008).
        self.q: dict[str, dict[tuple[str, str, int], float | None]] = {
            "spark": {},
            "wonder": {},
            "anchor": {},
        }
        self.verdicts: dict[str, dict[tuple[str, str, int], tuple[str, str]]] = {
            "spark": {},
            "wonder": {},
            "anchor": {},
        }
        self.anchors: dict[str, IdeaCard | None] = {}
        self.d_llm: dict[tuple[str, str, int, int], float | None] = {}
        self.d_emb: dict[tuple[str, str, int, int], float] = {}
        self.ground: dict[tuple[str, str, int], float | None] = {}
        self.progress = defaultdict(int)
        self.content_filtered: list[dict[str, Any]] = []
        self.conditions = tuple(cfg.get("conditions", CONDITIONS))

    def prompt(self, name: str) -> Prompt:
        """The version of a prompt this run's config names, e.g. "quality" -> quality-v2."""
        found = [p for p in self.cfg["prompts"] if p.startswith(f"{name}-")]
        if len(found) != 1:
            raise SystemExit(f"config must name exactly one {name} prompt, got {found}")
        return PROMPTS[found[0]]

    # --- setup ----------------------------------------------------------------

    async def check_models(self) -> None:
        models = await self.client("meta")._openai().models.list()
        available = {m.id for m in models.data}
        wanted = {self.cfg["generator"]["model"]} | {
            j["model"] for j in self.cfg["judges"].values()
        }
        missing = sorted(wanted - available)
        if missing:
            self.notes["models_missing"] = missing
            raise MissingModels(f"models not in /v1/models: {missing}")

    def instances(self) -> list[Instance]:
        a = load_instances(self.cfg["split"])
        picked = pick_baseline(a, len(self.cfg["instances"]))
        if picked != sorted(self.cfg["instances"]):
            raise SystemExit(f"config instances differ from pick_baseline(A): {picked}")
        by_id = {x.id: x for x in a}
        return [by_id[i] for i in sorted(self.cfg["instances"])]

    # --- generation -----------------------------------------------------------

    async def generate_call(
        self,
        client,
        x: Instance,
        condition: str,
        k: int,
        sample_idx: int,
        previous=(),
        *,
        card: Card | None = None,
        params: dict[str, Any] | None = None,
    ):
        gen = self.cfg["generator"]
        ideation = self.prompt("ideation")
        messages = ideation_messages(x, k, ideation, previous, card.text if card else None)
        prompt_hash = f"{ideation.hash}+{self.prompt('inject').hash}" if card else ideation.hash
        extra = {"instance": x.id, "condition": condition}
        attempt = 0
        try:
            c = await self.call(f"gen {condition}", gen, messages, params, sample_idx,
                                client=client, extra=extra)  # fmt: skip
            parsed = parse_set(c.content, k)
            if parsed.retryable:  # ADR-006: one retry, json_error only
                attempt = 1
                c = await self.call(f"gen {condition} retry", gen, messages, params, sample_idx,
                                    attempt=1, client=client, extra=extra)  # fmt: skip
                parsed = parse_set(c.content, k)
            call_key, usage = c.key, c.usage
        except ContentFilteredError as e:
            # Invalid, q = 0, reported apart; the run goes on (ADR-006, amendment 6).
            self.content_filtered.append({"judge": "generation", "attempt": attempt, **extra})
            parsed, call_key, usage = filtered_set(k), e.key, Usage()
        records = records_from_set(
            parsed,
            cards=[card.id if card else None] * k,
            run_id=self.run_id,
            instance=x.id,
            condition=condition,
            model=gen["model"],
            prompt_hash=prompt_hash,
            seed=None,
            call_key=call_key,
            usage=usage,
            attempt=attempt,
        )
        self.progress["generations"] += 1
        return records

    async def generate_sequence(self, client, x: Instance, condition: str):
        """Sequential set (ADR-009): K = 1 per call, conditioned on the valid ideas so far."""
        records, previous = [], []
        for slot in range(K):
            sample_idx = SEQUENTIAL_SAMPLE_BASE + slot
            r = (await self.generate_call(client, x, condition, 1, sample_idx, previous))[0]
            records.append(r.model_copy(update={"slot": slot}))
            if r.valid:
                previous = [*previous, r.idea]
        return records

    async def generate(self, instances: list[Instance]) -> None:
        gen = self.cfg["generator"]
        client = self.new_client(gen["provider"], concurrency=gen["concurrency"])

        async def c0(x):
            sets = await asyncio.gather(
                *(self.generate_call(client, x, "C0", 1, s) for s in range(K))
            )
            return [r[0].model_copy(update={"slot": s}) for s, r in enumerate(sets)]

        async def c1(x):
            return await self.generate_call(client, x, "C1", K, 0)

        async def c1s(x):
            return await self.generate_sequence(client, x, "C1s")

        makers = {"C0": c0, "C1": c1, "C1s": c1s}
        jobs = {(x.id, c): makers[c](x) for c in self.conditions for x in instances}
        results = await asyncio.gather(*jobs.values())
        self.slots = dict(zip(jobs, results, strict=True))
        write_jsonl(
            self.out / "generations.jsonl",
            [r for key in sorted(self.slots) for r in self.slots[key]],
        )
        self.notes["generator_client"] = vars(client.stats) | {"usage": vars(client.stats.usage)}

    async def generate_anchors(self, client, instances: list[Instance]) -> None:
        """One K = 1 idea per instance at the reserved sample_idx (ADR-008). An instance whose
        anchor is invalid has no primary q: its sets count as incomplete."""
        results = await asyncio.gather(
            *(self.generate_call(client, x, "ANCHOR", 1, ANCHOR_SAMPLE_IDX) for x in instances)
        )
        records = [r[0] for r in results]
        self.anchors = {x.id: r.idea for x, r in zip(instances, records, strict=True)}
        write_jsonl(self.out / "anchors.jsonl", records)

    # --- judges ---------------------------------------------------------------

    async def judge_call(self, client, judge: str, prompt, mode, messages, parse, ids: dict):
        """One judge call with one retry on a broken output contract.

        A request the provider's content filter refuses twice is a missing value, not a crash
        (ADR-006, amendment 6): it is logged and counted, never reworded.
        """
        for attempt in (0, 1):
            try:
                c = await self.call(
                    judge, mode, messages, attempt=attempt, client=client, extra=ids
                )
            except ContentFilteredError:
                self.content_filtered.append({"judge": judge, **ids})
                self.judge_rows.append(
                    {
                        "run_id": self.run_id,
                        "card": None,
                        "judge": judge,
                        "model": mode["model"],
                        "prompt_hash": prompt.hash,
                        "seed": None,
                        "attempt": attempt,
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "call_key": None,
                        "raw": None,
                        "parsed": None,
                        "parse_error": "content_filter",
                        **ids,
                    }
                )
                return None
            try:
                value, error = parse(c.content), None
            except JudgeOutputError as e:
                value, error = None, str(e)
            self.judge_rows.append(
                {
                    "run_id": self.run_id,
                    "card": None,
                    "judge": judge,
                    "model": c.model,
                    "prompt_hash": prompt.hash,
                    "seed": None,
                    "attempt": attempt,
                    "tokens_in": c.usage.tokens_in,
                    "tokens_cached": c.usage.tokens_cached,
                    "tokens_out": c.usage.tokens_out,
                    "tokens_reasoning": c.usage.tokens_reasoning,
                    "call_key": c.key,
                    "raw": c.content,
                    "parsed": value,
                    "parse_error": error,
                    **ids,
                }
            )
            self.progress[judge] += 1
            if error is None:
                return value
        return None

    async def quality(self, client, name: str, x: Instance, cond: str, slot: int, idea: IdeaCard):
        mode = self.cfg["judges"]["spark" if name == "anchor" else name]
        ids = {"instance": x.id, "condition": cond, "slot": slot}
        key = (x.id, cond, slot)
        if name == "anchor":
            if self.anchors[x.id] is None:
                self.q[name][key] = None
                return
            reference = render_card(self.anchors[x.id])
        else:
            reference = x.gold
        quality_prompt = self.prompt("quality")
        winners = []
        for idea_first in (True, False):
            msgs = quality_messages(idea, reference, idea_first=idea_first, prompt=quality_prompt)
            winners.append(
                await self.judge_call(
                    client,
                    f"quality-{name}",
                    quality_prompt,
                    mode,
                    msgs,
                    parse_winner,
                    ids | {"idea_first": idea_first},
                )  # fmt: skip
            )
        if None in winners:
            self.q[name][key] = None
        else:
            self.q[name][key] = q_from_orders(*winners)
            self.verdicts[name][key] = tuple(winners)

    async def distance_pair(self, client, x: Instance, cond: str, i: int, j: int, a, b):
        mode = self.cfg["judges"]["spark"]
        ids = {"instance": x.id, "condition": cond, "slot": i, "other_slot": j}
        s = await self.judge_call(client, "distance-spark", DISTANCE_V1, mode,
                                  distance_messages(a, b), parse_similarity, ids)  # fmt: skip
        self.d_llm[(x.id, cond, i, j)] = None if s is None else distance(s)

    async def grounding_one(self, client, x: Instance, cond: str, slot: int, idea: IdeaCard):
        mode = self.cfg["judges"]["spark"]
        ids = {"instance": x.id, "condition": cond, "slot": slot}
        s = await self.judge_call(client, "grounding-spark", GROUNDING_V1, mode,
                                  grounding_messages(x, idea), parse_grounding, ids)  # fmt: skip
        self.ground[(x.id, cond, slot)] = None if s is None else grounding(s)

    def judge_client(self, name: str):
        mode = self.cfg["judges"][name]
        return self.new_client(
            mode["provider"],
            mode["concurrency"],
            adaptive=mode.get("adaptive", False),
            max_concurrency=mode.get("max_concurrency"),
        )

    def client_stats(self, client) -> dict[str, Any]:
        return (
            vars(client.stats)
            | {"usage": vars(client.stats.usage)}
            | {"final_concurrency_limit": client.limiter.limit}
        )

    async def judge(self, instances: list[Instance]) -> None:
        spark = self.judge_client("spark")
        wonder = self.judge_client("wonder")
        jobs = []
        for x in instances:
            for cond in self.conditions:
                valid = [(r.slot, r.idea) for r in self.slots[(x.id, cond)] if r.valid]
                for slot, idea in valid:
                    jobs.append(self.quality(spark, "spark", x, cond, slot, idea))
                    jobs.append(self.quality(wonder, "wonder", x, cond, slot, idea))
                    jobs.append(self.grounding_one(spark, x, cond, slot, idea))
                    if self.primary_q == "anchor":
                        jobs.append(self.quality(spark, "anchor", x, cond, slot, idea))
                for (i, a), (j, b) in itertools.combinations(valid, 2):
                    jobs.append(self.distance_pair(spark, x, cond, i, j, a, b))
        await asyncio.gather(*jobs)
        write_jsonl_dicts(self.out / "judgments.jsonl", self.judge_rows)
        self.notes["judge_clients"] = {
            "spark": self.client_stats(spark),
            "wonder": self.client_stats(wonder),
        }

    def embed(self) -> None:
        from oblique.judges.embed import MODEL, REVISION, Embedder, pair_distance

        keys = [(k, r.slot) for k in sorted(self.slots) for r in self.slots[k] if r.valid]
        ideas = [r.idea for k in sorted(self.slots) for r in self.slots[k] if r.valid]
        vectors = dict(zip(keys, (self.embedder or Embedder()).embed(ideas), strict=True))
        for iid, cond, i, j in self.d_llm:
            self.d_emb[(iid, cond, i, j)] = pair_distance(
                vectors[((iid, cond), i)], vectors[((iid, cond), j)]
            )
        self.notes["embeddings"] = {"model": MODEL, "revision": REVISION, "ideas": len(ideas)}

    # --- metrics --------------------------------------------------------------

    @property
    def primary_q(self) -> str:
        """Which q drives CU: "anchor" (ADR-008) or "spark" (against the gold, runs before it)."""
        return {"anchor": "anchor", "gold": "spark"}[self.cfg["q_reference"]]

    def comparisons(self) -> list[tuple[str, str]]:
        """Paired comparisons to report: every condition against C0, and C1s against C1."""
        pairs = [(c, "C0") for c in self.conditions if c != "C0" and "C0" in self.conditions]
        if {"C1", "C1s"} <= set(self.conditions):
            pairs.append(("C1s", "C1"))
        return pairs

    def q_report(self, name: str) -> dict[str, Any]:
        """q per condition (invalid ideas count as 0): CI over instances, ties, position."""
        out: dict[str, Any] = {}
        per_instance: dict[str, dict[str, float]] = {}
        for cond in self.conditions:
            means: dict[str, float] = {}
            judged: list[float] = []
            verdicts: list[tuple[str, str]] = []
            for (iid, c), records in sorted(self.slots.items()):
                if c != cond:
                    continue
                keys = [(iid, c, r.slot) for r in records]
                qs = [
                    self.q[name].get(k) if r.valid else 0.0
                    for k, r in zip(keys, records, strict=True)
                ]
                judged += [self.q[name][k] for k, r in zip(keys, records, strict=True)
                           if r.valid and self.q[name].get(k) is not None]  # fmt: skip
                verdicts += [self.verdicts[name][k] for k in keys if k in self.verdicts[name]]
                if all(v is not None for v in qs):
                    means[iid] = sum(qs) / len(qs)
            per_instance[cond] = means
            out[cond] = {
                "q": ci_dict(bootstrap_mean_ci(list(means.values()))) if means else None,
                "ideas_judged": len(judged),
                "tie_share": judged.count(0.5) / len(judged) if judged else None,
                "win_share": judged.count(1.0) / len(judged) if judged else None,
                "loss_share": judged.count(0.0) / len(judged) if judged else None,
                "position": position_consistency(verdicts) if verdicts else None,
                "by_instance": means,
            }
        for a, b in self.comparisons():
            common = set(per_instance[a]) & set(per_instance[b])
            if common:
                out[f"q {a}-{b}"] = ci_dict(
                    paired_delta_ci(
                        {i: per_instance[a][i] for i in common},
                        {i: per_instance[b][i] for i in common},
                    )
                )
        verdicts = list(self.verdicts[name].values())
        out["position_all"] = position_consistency(verdicts) if verdicts else None
        return out

    def length_bias(self, name: str, instances: list[Instance]) -> dict[str, Any]:
        """Length in words (ADR-009) of the idea the judge reads vs its q, and vs the reference."""
        ideas = {(k[0], k[1], r.slot): r.idea for k, recs in self.slots.items() for r in recs}
        reference = {
            x.id: judged_words(self.anchors[x.id]) if name == "anchor" else len(x.gold.split())
            for x in instances
            if name != "anchor" or self.anchors.get(x.id) is not None
        }
        keys = [k for k, v in self.q[name].items() if v is not None]
        lengths = [judged_words(ideas[k]) for k in keys]
        rel = [judged_words(ideas[k]) - reference[k[0]] for k in keys]
        qs = [self.q[name][k] for k in keys]
        return {
            "n": len(keys),
            "pearson_len_q": pearson(lengths, qs),
            "spearman_len_q": spearman(lengths, qs),
            "pearson_len_minus_reference_q": pearson(rel, qs),
            "mean_idea_words": sum(lengths) / len(lengths) if lengths else None,
            "mean_reference_words": sum(reference.values()) / len(reference) if reference else None,
        }

    def set_metrics(self, key: SetKey) -> dict[str, Any]:
        iid, cond = key
        records = self.slots[key]
        q = [
            self.q[self.primary_q].get((iid, cond, r.slot), 0.0) if r.valid else 0.0
            for r in records
        ]
        complete = all(v is not None for v in q)
        D = [[0.0 if i == j else math.nan for j in range(K)] for i in range(K)]
        De = [row[:] for row in D]
        pairs = []
        for (a, c, i, j), d in self.d_llm.items():
            if (a, c) != key:
                continue
            if d is None:
                complete = False
                continue
            D[i][j] = D[j][i] = d
            De[i][j] = De[j][i] = self.d_emb[(a, c, i, j)]
            pairs.append((d, De[i][j]))
        out: dict[str, Any] = {
            "instance": iid,
            "condition": cond,
            "invalid": sum(not r.valid for r in records),
            "complete": complete,
        }
        if not complete:
            return out
        q = [float(v) for v in q]
        conforming = [r.conforming for r in records]
        g = [self.ground.get((iid, cond, r.slot)) for r in records if r.valid]
        q_gold = [self.q["spark"].get((iid, cond, r.slot)) if r.valid else 0.0 for r in records]
        out |= {
            "q": sum(q) / K,
            "q_gold": sum(q_gold) / K if all(v is not None for v in q_gold) else None,
            "cu": creative_utility(q, D),
            "cu_strict": creative_utility(q, D, strict=True, conforming=conforming),
            "cu_embed": creative_utility(q, De),
            "d": sum(p[0] for p in pairs) / len(pairs) if pairs else None,
            "d_embed": sum(p[1] for p in pairs) / len(pairs) if pairs else None,
            "grounding": sum(v for v in g if v is not None) / len(g) if g else None,
        }
        return out

    def metrics(self, instances: list[Instance]) -> dict[str, Any]:
        rows = [self.set_metrics(k) for k in sorted(self.slots)]
        complete = [r for r in rows if r["complete"]]
        ok_ids = set.intersection(
            *({r["instance"] for r in complete if r["condition"] == c} for c in self.conditions)
        )
        by = {
            c: {r["instance"]: r for r in complete if r["condition"] == c} for c in self.conditions
        }
        table = {}
        for cond in self.conditions:
            cell = {"n_sets": len(by[cond]), "invalid_ideas": 0, "ideas": 0}
            for r in rows:
                if r["condition"] == cond:
                    cell["invalid_ideas"] += r["invalid"]
                    cell["ideas"] += K
            for m in ("cu", "q", "q_gold", "d", "d_embed", "grounding", "cu_strict", "cu_embed"):
                vals = [by[cond][i][m] for i in sorted(ok_ids) if by[cond][i].get(m) is not None]
                cell[m] = ci_dict(bootstrap_mean_ci(vals)) if vals else None
            wonder_q = [
                v for (i, c, _), v in self.q["wonder"].items() if c == cond and v is not None
            ]
            cell["q_wonder_ideas_mean"] = sum(wonder_q) / len(wonder_q) if wonder_q else None
            table[cond] = cell
        deltas = {}
        for m, (x, y) in itertools.product(
            ("cu", "q", "q_gold", "d", "d_embed"), self.comparisons()
        ):
            a = {i: by[x][i][m] for i in ok_ids if by[x][i].get(m) is not None}
            b = {i: by[y][i][m] for i in ok_ids if by[y][i].get(m) is not None}
            common = set(a) & set(b)
            if common:
                ci = paired_delta_ci({i: a[i] for i in common}, {i: b[i] for i in common})
                deltas[f"{m} {x}-{y}"] = ci_dict(ci)
        return {
            "sets": rows,
            "paired_instances": len(ok_ids),
            "incomplete_sets": len(rows) - len(complete),
            "table": table,
            "deltas": deltas,
            "q_report": {n: self.q_report(n) for n, values in self.q.items() if values},
            "calibration": self.calibration(instances),
        }

    def calibration(self, instances: list[Instance]) -> dict[str, Any]:
        pairs = [(d, self.d_emb[k]) for k, d in self.d_llm.items() if d is not None]
        both = [
            k
            for k, v in self.q["spark"].items()
            if v is not None and self.q["wonder"].get(k) is not None
        ]
        orders = [
            (self.verdicts["spark"][k][o], self.verdicts["wonder"][k][o])
            for k in both
            for o in (0, 1)
        ]
        out: dict[str, Any] = {
            "spearman_d_llm_vs_embed": {
                "n": len(pairs),
                "rho": spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None,
            },
            "spark_vs_wonder_q": agreement(
                [self.q["spark"][k] for k in both], [self.q["wonder"][k] for k in both]
            )
            if both
            else None,
            "spark_vs_wonder_per_order": agreement([o[0] for o in orders], [o[1] for o in orders])
            if orders
            else None,
        }
        for name, values in self.q.items():
            if values:
                verdicts = list(self.verdicts[name].values())
                out[f"position_{name}"] = position_consistency(verdicts) if verdicts else None
                out[f"length_bias_{name}"] = self.length_bias(name, instances)
        return out

    def examples(self, instances: list[Instance]) -> dict[str, list[dict[str, Any]]]:
        """First two config instances (sorted IDs), first valid slot: no picking by result."""
        out = {}
        for cond in self.conditions:
            rows = []
            for x in instances[:2]:
                r = next((r for r in self.slots[(x.id, cond)] if r.valid), None)
                if r is None:
                    continue
                key = (x.id, cond, r.slot)
                rows.append(
                    {
                        "instance": x.id,
                        "domain": x.domain,
                        "slot": r.slot,
                        "idea": r.idea.model_dump(),
                        "q_spark": self.q["spark"].get(key),
                        "q_wonder": self.q["wonder"].get(key),
                        "grounding": self.ground.get(key),
                    }
                )
            out[cond] = rows
        return out

    # --- run ------------------------------------------------------------------

    async def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), "utf-8")
        try:
            await self.check_models()
            instances = self.instances()
            await self.generate(instances)
            if self.primary_q == "anchor":
                gen = self.cfg["generator"]
                client = self.new_client(gen["provider"], gen["concurrency"])
                await self.generate_anchors(client, instances)
            print("generation done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)
            await self.judge(instances)
            print("judges done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)
            self.embed()
            result = {
                "run_id": self.run_id,
                "spent_usd": round(self.spent_usd, 4),
                "metrics": self.metrics(instances),
                "examples": self.examples(instances),
                "notes": self.notes,
            }
            (self.out / "metrics.json").write_text(
                json.dumps(result, indent=2, default=str, ensure_ascii=False), "utf-8"
            )
        finally:
            write_jsonl_dicts(self.out / "calls.jsonl", self.calls)
            print(f"spent US${self.spent_usd:.4f} in {len(self.calls)} calls")


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.baseline configs/R-*.yaml")
    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    asyncio.run(Baseline(cfg).run())


if __name__ == "__main__":
    main(sys.argv[1:])
