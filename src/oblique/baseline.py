"""Baseline C0 / C1 on split A with judge calibration (R-20261030-03).

    uv run --group embed python -m oblique.baseline configs/R-20261030-03.yaml

C0: three independent single-idea samples (K = 1, sample_idx 0-2) form a set.
C1: one call that asks for a set of K = 3. No card, no seed.

Validity follows ADR-006: one retry (attempt 1) only on `json_error`; invalid
ideas get q = 0 and are not judged. Judges (author, 2026-10-30):
- quality, Spark, both orders against the gold (primary q);
- quality, Wonder, both orders (agreement only);
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
from oblique.generate.records import GenerationRecord, records_from_set, write_jsonl
from oblique.generate.schema import IdeaCard
from oblique.generate.validate import parse_set
from oblique.metrics import bootstrap_mean_ci, creative_utility, paired_delta_ci
from oblique.metrics.calibration import agreement, pearson, position_consistency, spearman
from oblique.prompts import (
    DISTANCE_V1,
    GROUNDING_V1,
    IDEATION_V2,
    QUALITY_V1,
    JudgeOutputError,
    distance,
    distance_messages,
    grounding,
    grounding_messages,
    ideation_messages,
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
SetKey = tuple[str, str]  # (instance, condition)


def ci_dict(ci) -> dict[str, float | int]:
    return {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


class Baseline(Smoke):
    def __init__(self, cfg: dict[str, Any], *, embedder: Any = None, **kw: Any):
        super().__init__(cfg, **kw)
        self.embedder = embedder  # tests only: stands in for the sentence-transformers model
        self.slots: dict[SetKey, list[GenerationRecord]] = {}
        self.judge_rows: list[dict[str, Any]] = []
        self.q: dict[str, dict[tuple[str, str, int], float | None]] = {"spark": {}, "wonder": {}}
        self.verdicts: dict[str, dict[tuple[str, str, int], tuple[str, str]]] = {
            "spark": {},
            "wonder": {},
        }
        self.d_llm: dict[tuple[str, str, int, int], float | None] = {}
        self.d_emb: dict[tuple[str, str, int, int], float] = {}
        self.ground: dict[tuple[str, str, int], float | None] = {}
        self.progress = defaultdict(int)

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

    async def generate_call(self, client, x: Instance, condition: str, k: int, sample_idx: int):
        gen = self.cfg["generator"]
        messages = ideation_messages(x, k)
        extra = {"instance": x.id, "condition": condition}
        c = await self.call(f"gen {condition}", gen, messages, sample_idx=sample_idx,
                            client=client, extra=extra)  # fmt: skip
        parsed, attempt = parse_set(c.content, k), 0
        if parsed.retryable:  # ADR-006: one retry, json_error only
            attempt = 1
            c = await self.call(f"gen {condition} retry", gen, messages, sample_idx=sample_idx,
                                attempt=1, client=client, extra=extra)  # fmt: skip
            parsed = parse_set(c.content, k)
        records = records_from_set(
            parsed,
            cards=[None] * k,
            run_id=self.run_id,
            instance=x.id,
            condition=condition,
            model=c.model,
            prompt_hash=IDEATION_V2.hash,
            seed=None,
            call_key=c.key,
            usage=c.usage,
            attempt=attempt,
        )
        self.progress["generations"] += 1
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

        jobs = {(x.id, "C0"): c0(x) for x in instances} | {(x.id, "C1"): c1(x) for x in instances}
        results = await asyncio.gather(*jobs.values())
        self.slots = dict(zip(jobs, results, strict=True))
        write_jsonl(
            self.out / "generations.jsonl",
            [r for key in sorted(self.slots) for r in self.slots[key]],
        )
        self.notes["generator_client"] = vars(client.stats) | {"usage": vars(client.stats.usage)}

    # --- judges ---------------------------------------------------------------

    async def judge_call(self, client, judge: str, prompt, mode, messages, parse, ids: dict):
        """One judge call with one retry on a broken output contract."""
        for attempt in (0, 1):
            c = await self.call(judge, mode, messages, attempt=attempt, client=client, extra=ids)
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
        mode = self.cfg["judges"][name]
        ids = {"instance": x.id, "condition": cond, "slot": slot}
        winners = []
        for idea_first in (True, False):
            msgs = quality_messages(idea, x.gold, idea_first=idea_first)
            winners.append(
                await self.judge_call(
                    client,
                    f"quality-{name}",
                    QUALITY_V1,
                    mode,
                    msgs,
                    parse_winner,
                    ids | {"idea_first": idea_first},
                )  # fmt: skip
            )
        key = (x.id, cond, slot)
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

    async def judge(self, instances: list[Instance]) -> None:
        judges = self.cfg["judges"]
        spark = self.new_client(judges["spark"]["provider"], judges["spark"]["concurrency"])
        wonder = self.new_client(judges["wonder"]["provider"], judges["wonder"]["concurrency"])
        jobs = []
        for x in instances:
            for cond in CONDITIONS:
                valid = [(r.slot, r.idea) for r in self.slots[(x.id, cond)] if r.valid]
                for slot, idea in valid:
                    jobs.append(self.quality(spark, "spark", x, cond, slot, idea))
                    jobs.append(self.quality(wonder, "wonder", x, cond, slot, idea))
                    jobs.append(self.grounding_one(spark, x, cond, slot, idea))
                for (i, a), (j, b) in itertools.combinations(valid, 2):
                    jobs.append(self.distance_pair(spark, x, cond, i, j, a, b))
        await asyncio.gather(*jobs)
        write_jsonl_dicts(self.out / "judgments.jsonl", self.judge_rows)
        self.notes["judge_clients"] = {
            "spark": vars(spark.stats) | {"usage": vars(spark.stats.usage)},
            "wonder": vars(wonder.stats) | {"usage": vars(wonder.stats.usage)},
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

    def set_metrics(self, key: SetKey) -> dict[str, Any]:
        iid, cond = key
        records = self.slots[key]
        q = [self.q["spark"].get((iid, cond, r.slot), 0.0) if r.valid else 0.0 for r in records]
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
        out |= {
            "q": sum(q) / K,
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
        ok_ids = {r["instance"] for r in complete if r["condition"] == "C0"} & {
            r["instance"] for r in complete if r["condition"] == "C1"
        }
        by = {c: {r["instance"]: r for r in complete if r["condition"] == c} for c in CONDITIONS}
        table = {}
        for cond in CONDITIONS:
            cell = {"n_sets": len(by[cond]), "invalid_ideas": 0, "ideas": 0}
            for r in rows:
                if r["condition"] == cond:
                    cell["invalid_ideas"] += r["invalid"]
                    cell["ideas"] += K
            for m in ("cu", "q", "d", "d_embed", "grounding", "cu_strict", "cu_embed"):
                vals = [by[cond][i][m] for i in sorted(ok_ids) if by[cond][i].get(m) is not None]
                cell[m] = ci_dict(bootstrap_mean_ci(vals)) if vals else None
            wonder_q = [
                v for (i, c, _), v in self.q["wonder"].items() if c == cond and v is not None
            ]
            cell["q_wonder_ideas_mean"] = sum(wonder_q) / len(wonder_q) if wonder_q else None
            table[cond] = cell
        deltas = {}
        for m in ("cu", "q", "d", "d_embed"):
            a = {i: by["C1"][i][m] for i in ok_ids if by["C1"][i].get(m) is not None}
            b = {i: by["C0"][i][m] for i in ok_ids if by["C0"][i].get(m) is not None}
            common = set(a) & set(b)
            if common:
                ci = paired_delta_ci({i: a[i] for i in common}, {i: b[i] for i in common})
                deltas[f"{m} C1-C0"] = ci_dict(ci)
        return {
            "sets": rows,
            "paired_instances": len(ok_ids),
            "incomplete_sets": len(rows) - len(complete),
            "table": table,
            "deltas": deltas,
            "calibration": self.calibration(instances),
        }

    def calibration(self, instances: list[Instance]) -> dict[str, Any]:
        gold_len = {x.id: len(x.gold) for x in instances}
        ideas = {(k[0], k[1], r.slot): r.idea for k, recs in self.slots.items() for r in recs}
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
        for name in ("spark", "wonder"):
            verdicts = list(self.verdicts[name].values())
            out[f"position_{name}"] = position_consistency(verdicts) if verdicts else None
            keys = [k for k, v in self.q[name].items() if v is not None]
            lengths = [len(render_card(ideas[k])) for k in keys]
            rel = [len(render_card(ideas[k])) - gold_len[k[0]] for k in keys]
            qs = [self.q[name][k] for k in keys]
            out[f"length_bias_{name}"] = {
                "n": len(keys),
                "pearson_len_q": pearson(lengths, qs),
                "spearman_len_q": spearman(lengths, qs),
                "pearson_len_minus_gold_q": pearson(rel, qs),
                "mean_idea_chars": sum(lengths) / len(lengths) if lengths else None,
                "mean_gold_chars": sum(gold_len.values()) / len(gold_len),
            }
        return out

    def examples(self, instances: list[Instance]) -> dict[str, list[dict[str, Any]]]:
        """First two config instances (sorted IDs), first valid slot: no picking by result."""
        out = {}
        for cond in CONDITIONS:
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
