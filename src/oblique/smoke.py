"""Smoke test and cost gate (R-20261030-01): one split A instance, every call kind once.

    python -m oblique.smoke configs/R-20261030-01.yaml

Answers, before any sweep: does the generator return a parseable K=3 set; does
the API honour `seed`; do repeated prefixes come back as cached tokens; is JSON
mode accepted; what does the same prompt cost in Contributor vs Standard; do
the four prompts parse with Spark (and quality with Wonder); how many reasoning
tokens each call spends. Every call goes through the response cache, so a
rerun costs nothing. Outputs go to runs/<run_id>/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import openai
import yaml

from oblique.bench import load_instances
from oblique.cache import Cache
from oblique.generate.records import records_from_set, write_jsonl
from oblique.generate.validate import parse_set
from oblique.llm import PROVIDERS, Completion, LLMClient
from oblique.prompts import (
    DISTANCE_V1,
    GROUNDING_V1,
    IDEATION_V2,
    QUALITY_V1,
    JudgeOutputError,
    Prompt,
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
)

Messages = list[dict[str, Any]]


class BudgetExceeded(RuntimeError):
    pass


class MissingModels(SystemExit):
    pass


def load_env(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines from .env into os.environ, without overriding what is set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


class Smoke:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        out: Path | None = None,
        transport: Any = None,
        wait: Any = None,
    ):
        self.cfg = cfg
        self.run_id = cfg["run_id"]
        self.out = out or Path("runs") / self.run_id
        self.cache = Cache(cfg["cache"])
        self.transport = transport  # tests only: stands in for the OpenAI client
        self.wait = wait  # tests only: retry wait strategy (default: the client's backoff)
        self.clients: dict[str, LLMClient] = {}
        self.calls: list[dict[str, Any]] = []
        self.judgments: list[dict[str, Any]] = []
        self.notes: dict[str, Any] = {}
        self.spent_usd = 0.0

    def new_client(self, provider: str, concurrency: int = 1) -> LLMClient:
        return LLMClient(
            provider, self.cache, concurrency=concurrency, client=self.transport, wait=self.wait
        )

    def client(self, provider: str) -> LLMClient:
        if provider not in self.clients:
            self.clients[provider] = self.new_client(provider)
        return self.clients[provider]

    async def call(
        self,
        label: str,
        mode: dict[str, Any],
        messages: Messages,
        params: dict[str, Any] | None = None,
        sample_idx: int = 0,
        *,
        attempt: int = 0,
        client: LLMClient | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Completion:
        params = {**mode.get("params", {}), **(params or {})}
        if self.spent_usd > self.cfg["budget"]["stop_usd"]:
            raise BudgetExceeded(f"spent US${self.spent_usd:.3f}; stopping before {label!r}")
        t0 = time.perf_counter()
        c = await (client or self.client(mode["provider"])).complete(
            mode["model"], messages, params=params, sample_idx=sample_idx, attempt=attempt
        )
        cost = 0.0 if c.from_cache else self.cost(c)
        self.spent_usd += cost
        choice = (c.response.get("choices") or [{}])[0]
        self.calls.append(
            {
                "label": label,
                "model": c.model,
                "provider": mode["provider"],
                "params": params,
                "sample_idx": sample_idx,
                "attempt": attempt,
                "from_cache": c.from_cache,
                "seconds": round(time.perf_counter() - t0, 2),
                "finish_reason": choice.get("finish_reason"),
                "tokens_in": c.usage.tokens_in,
                "tokens_cached": c.usage.tokens_cached,
                "tokens_out": c.usage.tokens_out,
                "tokens_reasoning": c.usage.tokens_reasoning,
                "usage_raw": c.response.get("usage"),
                "cost_usd": round(cost, 5),
                "call_key": c.key,
                **(extra or {}),
            }
        )
        return c

    def cost(self, c: Completion) -> float:
        """Cost of one call at the configured prices (USD per million tokens).

        Reasoning tokens are already inside `completion_tokens` (verified in
        R-20261030-01), so they are not added again.
        """
        p = self.cfg["budget"]["prices_per_m"][c.model]
        u = c.usage
        uncached = u.tokens_in - u.tokens_cached
        return (uncached * p["in"] + u.tokens_cached * p["cached"] + u.tokens_out * p["out"]) / 1e6

    async def first_accepted(
        self,
        label: str,
        mode: dict[str, Any],
        messages: Messages,
        key: str,
        values: list[Any],
        base: dict[str, Any] | None = None,
        sample_idx: int = 0,
    ) -> tuple[Completion, dict[str, Any]]:
        """Try `params[key] = v` for each v in order (None: omit it); the first accepted wins.

        Returns the completion and the params that worked. Rejected requests (400)
        never reach the cache.
        """
        rejected = []
        for v in values:
            params = {**(base or {}), **({key: v} if v is not None else {})}
            try:
                c = await self.call(label, mode, messages, params, sample_idx)
            except openai.BadRequestError as e:
                rejected.append({"value": v, "error": str(e)[:300]})
                continue
            self.notes[f"{label}: {key}"] = {"accepted": v, "rejected": rejected}
            return c, params
        raise RuntimeError(f"{label}: no accepted value for {key}: {rejected}")

    def judged(
        self, judge: str, prompt: Prompt, c: Completion, parse: Callable[[str | None], Any], **extra
    ) -> Any:
        try:
            value, error = parse(c.content), None
        except JudgeOutputError as e:
            value, error = None, str(e)
        self.judgments.append(
            {
                "run_id": self.run_id,
                "instance": self.cfg["instance"],
                "card": None,
                "judge": judge,
                "model": c.model,
                "prompt_hash": prompt.hash,
                "seed": None,
                "tokens_in": c.usage.tokens_in,
                "tokens_out": c.usage.tokens_out,
                "tokens_reasoning": c.usage.tokens_reasoning,
                "call_key": c.key,
                "raw": c.content,
                "parsed": value,
                "parse_error": error,
                **extra,
            }
        )
        return value

    async def generate(self, instance) -> list[Completion]:
        k, gen = self.cfg["k"], self.cfg["generator"]
        messages = ideation_messages(instance, k)
        # JSON mode first; plain text if the API rejects it.
        g1, params = await self.first_accepted(
            "gen contributor seed=0",
            gen["contributor"],
            messages,
            "response_format",
            [{"type": "json_object"}, None],
            base={"seed": 0},
        )
        self.notes["json_mode_accepted"] = "response_format" in params
        # Identical API request; sample_idx only separates the local cache entries.
        g2 = await self.call("gen contributor seed=0 repeat", gen["contributor"], messages,
                             params, sample_idx=1)  # fmt: skip
        g3 = await self.call("gen contributor seed=1", gen["contributor"], messages,
                             {**params, "seed": 1})  # fmt: skip
        g4 = await self.call("gen standard seed=0", gen["standard"], messages, params)
        g5 = await self.call("gen standard seed=0 low", gen["standard_low"], messages, params)
        self.notes["seed"] = {
            "same_seed_same_text": g1.content == g2.content,
            "other_seed_same_text": g1.content == g3.content,
        }
        gens = (("g1", g1, 0), ("g2", g2, 0), ("g3", g3, 1), ("g4", g4, 0), ("g5", g5, 0))
        self.notes["generation reasoning tokens"] = {
            name: {"tokens_out": c.usage.tokens_out, "tokens_reasoning": c.usage.tokens_reasoning}
            for name, c, _ in gens
        }
        records = []
        for name, c, seed in gens:
            parsed = parse_set(c.content, k)
            self.notes[f"parse {name}"] = {
                "valid": len(parsed.valid),
                "shape": parsed.shape,
                "fenced": parsed.fenced,
                "invalid": [(s.reason, s.detail) for s in parsed.invalid],
            }
            records += records_from_set(
                parsed,
                cards=[None] * k,
                run_id=self.run_id,
                instance=instance.id,
                condition=f"smoke-{name}",
                model=c.model,
                prompt_hash=IDEATION_V2.hash,
                seed=seed,
                call_key=c.key,
                usage=c.usage,
            )
        write_jsonl(self.out / "generations.jsonl", records)
        return [c for _, c, _ in gens]

    async def judge(self, instance, idea, other) -> None:
        for name in ("spark", "wonder"):
            mode = self.cfg["judges"][name]
            efforts = mode.get("reasoning_effort", [None])
            msgs = quality_messages(idea, instance.gold, idea_first=True)
            c, params = await self.first_accepted(
                f"quality {name} idea_first", mode, msgs, "reasoning_effort", efforts
            )
            w1 = self.judged(f"quality-{name}", QUALITY_V1, c, parse_winner, idea_first=True)
            msgs = quality_messages(idea, instance.gold, idea_first=False)
            c = await self.call(f"quality {name} gold_first", mode, msgs, params)
            w2 = self.judged(f"quality-{name}", QUALITY_V1, c, parse_winner, idea_first=False)
            self.notes[f"quality {name}"] = {
                "idea_first": w1,
                "gold_first": w2,
                "q": q_from_orders(w1, w2) if w1 and w2 else None,
            }
            if name != "spark":
                continue
            c = await self.call("distance spark", mode, distance_messages(idea, other), params)
            s = self.judged("distance-spark", DISTANCE_V1, c, parse_similarity)
            self.notes["distance spark"] = {"similarity": s, "d": distance(s) if s else None}
            c = await self.call("grounding spark", mode, grounding_messages(instance, idea), params)
            g = self.judged("grounding-spark", GROUNDING_V1, c, parse_grounding)
            self.notes["grounding spark"] = {"score": g, "grounding": grounding(g) if g else None}

    async def run(self) -> None:
        cfg = self.cfg
        contributor, standard = cfg["generator"]["contributor"], cfg["generator"]["standard"]
        if (contributor["model"], contributor["params"]) == (standard["model"], standard["params"]):
            # The cache key has model and params but not the provider: the Standard call
            # would be a cache hit of the Contributor one and the price never compared.
            raise SystemExit("contributor and standard modes are identical; configure the mode")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")
        instance = next(x for x in load_instances(cfg["split"]) if x.id == cfg["instance"])
        # GET /v1/models (free) before any paid call: every configured model must exist.
        models = await self.client("meta")._openai().models.list()
        available = sorted(m.id for m in models.data)
        self.notes["models_available"] = available
        wanted = {m["model"] for m in [*cfg["generator"].values(), *cfg["judges"].values()]}
        missing = sorted(wanted - set(available))
        if missing:
            self.notes["models_missing"] = missing
            self.write()
            raise MissingModels(f"models not in /v1/models: {missing}; available: {available}")
        try:
            g1 = (await self.generate(instance))[0]
            ideas = parse_set(g1.content, cfg["k"]).valid
            if len(ideas) >= 2:
                await self.judge(instance, ideas[0], ideas[1])
            else:
                self.notes["judges"] = "skipped: fewer than 2 valid ideas in g1"
        finally:
            self.write()

    def write(self) -> None:
        write_jsonl_dicts(self.out / "calls.jsonl", self.calls)
        write_jsonl_dicts(self.out / "judgments.jsonl", self.judgments)
        stats = {
            p: {k: v for k, v in vars(c.stats).items() if k != "usage"} | vars(c.stats.usage)
            for p, c in self.clients.items()
        }
        summary = {
            "run_id": self.run_id,
            "spent_usd": round(self.spent_usd, 4),
            "notes": self.notes,
            "client_stats": stats,
        }
        (self.out / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=str, ensure_ascii=False), "utf-8"
        )
        print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
        for c in self.calls:
            print(
                f"{c['label']:<32} in={c['tokens_in']:>6} cached={c['tokens_cached']:>6} "
                f"out={c['tokens_out']:>6} reasoning={c['tokens_reasoning']:>6} "
                f"finish={c['finish_reason']} cache_hit={c['from_cache']}"
            )


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.smoke configs/R-*.yaml")
    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    modes = [*cfg["generator"].values(), *cfg["judges"].values()]
    unknown = {m["provider"] for m in modes} - set(PROVIDERS)
    if unknown:
        raise SystemExit(f"unknown providers {sorted(unknown)}")
    asyncio.run(Smoke(cfg).run())


if __name__ == "__main__":
    main(sys.argv[1:])
