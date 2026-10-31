"""Cost gate before the baseline (R-20261030-02). Generator calls only, no judges.

    python -m oblique.gate configs/R-20261030-02.yaml

1. Provider prompt cache: two calls on one instance with the same prefix
   (system + papers) and a different tail (K = 3, then K = 1), back to back.
2. Highest temperature the generator accepts, trying from `max` down; then
   whether the JSON still parses at that temperature (`samples` calls).
3. Concurrency: for each level N, `waves` sequential waves of N simultaneous
   calls on N different split A instances, each wave with a fresh client, so
   429s, retries, failures and latency are counted per wave.

No `seed` is sent (the API ignores it, R-20261030-01). Gate calls use
sample_idx >= 9000 so they never share a cache key with experiment calls.
Every call goes through the response cache; a rerun costs nothing.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

import openai
import yaml

from oblique.bench import load_instances
from oblique.generate.validate import parse_set
from oblique.prompts import IDEATION_V2, ideation_messages
from oblique.smoke import BudgetExceeded, MissingModels, Smoke, load_env

GATE_SAMPLE_BASE = 9000


def latency_summary(seconds: list[float]) -> dict[str, float | None]:
    if not seconds:
        return {"p50": None, "p90": None, "max": None}
    ordered = sorted(seconds)
    p90 = ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))]
    return {
        "p50": round(statistics.median(ordered), 1),
        "p90": round(p90, 1),
        "max": round(ordered[-1], 1),
    }


class Gate(Smoke):
    async def check_models(self) -> None:
        models = await self.client("meta")._openai().models.list()
        available = sorted(m.id for m in models.data)
        missing = sorted({self.cfg["generator"]["model"]} - set(available))
        if missing:
            self.notes["models_missing"] = missing
            raise MissingModels(f"models not in /v1/models: {missing}")

    async def cache_test(self, instance) -> None:
        gen = self.cfg["generator"]
        base = GATE_SAMPLE_BASE
        first = await self.call(
            "cache k=3", gen, ideation_messages(instance, 3, IDEATION_V2), sample_idx=base
        )
        second = await self.call(
            "cache k=1 (same prefix)",
            gen,
            ideation_messages(instance, 1, IDEATION_V2),
            sample_idx=base,
        )
        self.notes["cache"] = {
            "instance": instance.id,
            "first": {"in": first.usage.tokens_in, "cached": first.usage.tokens_cached},
            "second": {"in": second.usage.tokens_in, "cached": second.usage.tokens_cached},
            "second_cached_share": round(second.usage.tokens_cached / second.usage.tokens_in, 3)
            if second.usage.tokens_in
            else None,
        }

    async def temperature_test(self, instance) -> None:
        gen, cfg = self.cfg["generator"], self.cfg["temperature"]
        candidates = [round(cfg["max"] - i * cfg["step"], 2) for i in range(cfg["tries"])]
        messages = ideation_messages(instance, 3, IDEATION_V2)
        c, params = await self.first_accepted(
            "temperature probe",
            gen,
            messages,
            "temperature",
            candidates,
            sample_idx=GATE_SAMPLE_BASE,
        )
        results = [c]
        for j in range(1, cfg["samples"]):
            results.append(
                await self.call(
                    f"temperature {params['temperature']} sample {j}",
                    gen,
                    messages,
                    params,
                    sample_idx=GATE_SAMPLE_BASE + j,
                )
            )
        parsed = [parse_set(r.content, 3) for r in results]
        self.notes["temperature"] = {
            "max_accepted": params["temperature"],
            "parses": [
                {"valid": len(p.valid), "shape": p.shape, "reasons": [s.reason for s in p.invalid]}
                for p in parsed
            ],
        }

    async def concurrency_test(self, instances) -> None:
        gen, cfg = self.cfg["generator"], self.cfg["concurrency"]
        rows = []
        walls: dict[tuple[int, int], float] = {}
        for level in cfg["levels"]:
            for wave in range(cfg["waves"]):
                client = self.new_client(gen["provider"], concurrency=level)
                sample_idx = GATE_SAMPLE_BASE + 100 * level + wave

                async def one(x, client=client, level=level, wave=wave, sample_idx=sample_idx):
                    t0 = time.perf_counter()
                    try:
                        c = await self.call(
                            f"concurrency {level} wave {wave}",
                            gen,
                            ideation_messages(x, 3, IDEATION_V2),
                            sample_idx=sample_idx,
                            client=client,
                            extra={"level": level, "wave": wave, "instance": x.id},
                        )
                    except (openai.APIError, BudgetExceeded) as e:
                        return None, time.perf_counter() - t0, type(e).__name__
                    return c, time.perf_counter() - t0, None

                batch = [instances[(wave * level + j) % len(instances)] for j in range(level)]
                t0 = time.perf_counter()
                results = await asyncio.gather(*(one(x) for x in batch))
                wall = walls[(level, wave)] = time.perf_counter() - t0
                ok = [s for c, s, _ in results if c is not None and not c.from_cache]
                errors = [err for _, _, err in results if err]
                sets = sum(
                    len(parse_set(c.content, 3).valid) == 3 for c, _, _ in results if c is not None
                )
                rows.append(
                    {
                        "level": level,
                        "wave": wave,
                        "requests": level,
                        "ok": len(ok),
                        "failed": len(errors),
                        "errors": sorted(set(errors)),
                        "rate_limited_429": client.stats.rate_limited,
                        "retries": client.stats.retries,
                        "wall_s": round(wall, 1),
                        "latency_s": latency_summary(ok),
                        "valid_sets": sets,
                        "sets_per_min": round(60 * sets / wall, 1) if wall else None,
                    }
                )
                print(rows[-1])
                if "BudgetExceeded" in errors:
                    self.notes["concurrency"] = rows
                    raise BudgetExceeded("budget reached during the concurrency test")
        self.notes["concurrency"] = rows
        top = [r for r in rows if r["level"] == max(cfg["levels"])]
        top_wall = sum(walls[(r["level"], r["wave"])] for r in top)
        self.notes["top_level_sets_per_min"] = (
            round(60 * sum(r["valid_sets"] for r in top) / top_wall, 1) if top_wall else None
        )
        # Sustained = 429s in every wave at the top level, or any request that still failed.
        sustained = all(r["rate_limited_429"] > 0 for r in top) or any(r["failed"] for r in rows)
        self.notes["contributor_holds_top_level"] = not sustained

    async def run(self) -> None:
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")
        instances = load_instances(cfg["split"])
        by_id = {x.id: x for x in instances}
        try:
            await self.check_models()
            await self.cache_test(by_id[cfg["cache_test"]["instance"]])
            await self.temperature_test(by_id[cfg["temperature"]["instance"]])
            await self.concurrency_test(instances)
        finally:
            self.write()


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.gate configs/R-*.yaml")
    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    asyncio.run(Gate(cfg).run())


if __name__ == "__main__":
    main(sys.argv[1:])
