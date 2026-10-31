"""Validate the anchor as the quality reference (ADR-008, R-20261030-04).

    python -m oblique.anchor_check configs/R-20261030-04.yaml

Generates the anchor idea of each baseline instance and re-judges the existing
C0 / C1 ideas (`source_generations`) against it with quality-v1, both orders.
No new C0 / C1 ideas. Reports q with a CI per condition, the share of ties
(q = 0.5) and position consistency. A tie happens only when the same position
wins both orders, so the tie share equals 1 - consistency by construction.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from oblique.baseline import Baseline, K
from oblique.generate.records import read_generations
from oblique.smoke import load_env, write_jsonl_dicts


class AnchorCheck(Baseline):
    def load_source(self, instances) -> None:
        """The source run's records, checked against this run's instances and prompt."""
        records = read_generations(self.cfg["source_generations"])
        grouped = defaultdict(list)
        for r in records:
            if r.prompt_hash != self.prompt("ideation").hash:
                raise SystemExit(f"{r.instance}: idea from another prompt ({r.prompt_hash})")
            grouped[(r.instance, r.condition)].append(r)
        wanted = {(x.id, c) for x in instances for c in self.conditions}
        if set(grouped) != wanted or any(len(v) != K for v in grouped.values()):
            raise SystemExit("source generations do not match the configured instances")
        self.slots = {k: sorted(v, key=lambda r: r.slot) for k, v in grouped.items()}

    async def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), "utf-8")
        try:
            await self.check_models()
            instances = self.instances()
            self.load_source(instances)

            gen = self.cfg["generator"]
            generator = self.new_client(gen["provider"], gen["concurrency"])
            await self.generate_anchors(generator, instances)
            print("anchors done:", f"US${self.spent_usd:.3f}", flush=True)

            judge = self.judge_client("spark")
            by_id = {x.id: x for x in instances}
            jobs = [
                self.quality(judge, "anchor", by_id[iid], cond, r.slot, r.idea)
                for (iid, cond), records in sorted(self.slots.items())
                for r in records
                if r.valid
            ]
            await asyncio.gather(*jobs)
            result = {
                "run_id": self.run_id,
                "spent_usd": round(self.spent_usd, 4),
                "anchors_valid": sum(a is not None for a in self.anchors.values()),
                "q_anchor": self.q_report("anchor"),
                "length_bias_anchor": self.length_bias("anchor", instances),
                "clients": {
                    "generator": self.client_stats(generator),
                    "judge": self.client_stats(judge),
                },
                "notes": self.notes,
            }
            (self.out / "metrics.json").write_text(
                json.dumps(result, indent=2, default=str, ensure_ascii=False), "utf-8"
            )
        finally:
            write_jsonl_dicts(self.out / "judgments.jsonl", self.judge_rows)
            write_jsonl_dicts(self.out / "calls.jsonl", self.calls)
            print(f"spent US${self.spent_usd:.4f} in {len(self.calls)} calls")


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.anchor_check configs/R-*.yaml")
    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    asyncio.run(AnchorCheck(cfg).run())


if __name__ == "__main__":
    main(sys.argv[1:])
