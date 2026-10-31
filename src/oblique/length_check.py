"""Validate length control and sequential generation (ADR-009, R-20261030-05).

    uv run --group embed python -m oblique.length_check configs/R-20261030-05.yaml

New anchors and new C0, C1 and C1s ideas with ideation-v3, judged against the
anchor with quality-v2 in both orders, plus grounding. Reports length per
condition, r(length, q), q with CIs, position consistency and the win rate
controlled for length (ADR-009, point 7). Length is the words of gap + claim +
method, what the quality judge reads. Distance from local embeddings is a free
side check that sequential sets stay diverse; no distance judge runs here.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from oblique.baseline import Baseline, ci_dict
from oblique.metrics import bootstrap_mean_ci, paired_delta_ci
from oblique.metrics.calibration import controlled_win_rates, pearson, spearman
from oblique.prompts import WORD_LIMITS, field_words, judged_words
from oblique.smoke import load_env, write_jsonl_dicts

R_FALLBACK = 0.3  # ADR-009, point 7


def distribution(values: list[int]) -> dict[str, float | int]:
    p25, median, p75 = np.percentile(values, [25, 50, 75])
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": int(min(values)),
        "p25": float(p25),
        "median": float(median),
        "p75": float(p75),
        "max": int(max(values)),
    }


class LengthCheck(Baseline):
    def valid_ideas(self, cond: str | None = None):
        """((instance, condition, slot), idea) of every valid idea, optionally in one condition."""
        return [
            ((iid, c, r.slot), r.idea)
            for (iid, c), records in sorted(self.slots.items())
            for r in records
            if r.valid and (cond is None or c == cond)
        ]

    def within_caps(self, ideas: list) -> dict[str, float]:
        words = [field_words(idea) for idea in ideas]
        share = {f: sum(w[f] <= cap for w in words) / len(words) for f, cap in WORD_LIMITS.items()}
        share["all_fields"] = sum(
            all(w[f] <= cap for f, cap in WORD_LIMITS.items()) for w in words
        ) / len(words)
        return share

    def length_report(self) -> dict[str, Any]:
        anchors = [a for a in self.anchors.values() if a is not None]
        out: dict[str, Any] = {
            "ANCHOR": {
                "words": distribution([judged_words(a) for a in anchors]),
                "within_caps": self.within_caps(anchors),
            }
        }
        for cond in self.conditions:
            ideas = [idea for _, idea in self.valid_ideas(cond)]
            delta = [
                judged_words(idea) - judged_words(self.anchors[k[0]])
                for k, idea in self.valid_ideas(cond)
                if self.anchors[k[0]] is not None
            ]
            out[cond] = {
                "words": distribution([judged_words(i) for i in ideas]),
                "minus_anchor": distribution(delta),
                "within_caps": self.within_caps(ideas),
            }
        return out

    def length_vs_q(self) -> dict[str, Any]:
        def corr(keys_ideas) -> dict[str, float | int | None]:
            pairs = [(k, idea) for k, idea in keys_ideas if self.q["anchor"].get(k) is not None]
            qs = [self.q["anchor"][k] for k, _ in pairs]
            words = [judged_words(idea) for _, idea in pairs]
            delta = [
                w - judged_words(self.anchors[k[0]]) for w, (k, _) in zip(words, pairs, strict=True)
            ]
            return {
                "n": len(pairs),
                "pearson_words_q": pearson(words, qs),
                "spearman_words_q": spearman(words, qs),
                "pearson_minus_anchor_q": pearson(delta, qs),
            }

        out = {"all": corr(self.valid_ideas())}
        out |= {c: corr(self.valid_ideas(c)) for c in self.conditions}
        r = out["all"]["pearson_words_q"]
        out["fallback_triggered"] = r is not None and r >= R_FALLBACK
        out["fallback_threshold"] = R_FALLBACK
        return out

    def controlled(self, n_boot: int) -> dict[str, Any]:
        rows = []
        for k, idea in self.valid_ideas():
            if k not in self.verdicts["anchor"]:
                continue
            delta = judged_words(idea) - judged_words(self.anchors[k[0]])
            idea_first, gold_first = self.verdicts["anchor"][k]
            for win in (idea_first == "A", gold_first == "B"):
                rows.append({"instance": k[0], "condition": k[1], "win": win, "delta_words": delta})
        return controlled_win_rates(
            rows, self.conditions, comparisons=self.comparisons(), n_boot=n_boot
        )

    def grounding_report(self) -> dict[str, Any]:
        per: dict[str, dict[str, float]] = {}
        for cond in self.conditions:
            values: dict[str, list[float]] = {}
            for (iid, c, _), g in self.ground.items():
                if c == cond and g is not None:
                    values.setdefault(iid, []).append(g)
            per[cond] = {i: sum(v) / len(v) for i, v in values.items()}
        out: dict[str, Any] = {
            c: ci_dict(bootstrap_mean_ci(list(v.values()))) for c, v in per.items()
        }
        for a, b in self.comparisons():
            common = set(per[a]) & set(per[b])
            out[f"{a}-{b}"] = ci_dict(
                paired_delta_ci({i: per[a][i] for i in common}, {i: per[b][i] for i in common})
            )
        return out

    def embedding_distance(self) -> dict[str, Any]:
        """Mean pairwise embedding distance per set: free, and only a side check."""
        from oblique.judges.embed import MODEL, REVISION, Embedder, pair_distance

        keyed = self.valid_ideas()
        vectors = dict(
            zip(
                [k for k, _ in keyed],
                (self.embedder or Embedder()).embed([idea for _, idea in keyed]),
                strict=True,
            )
        )
        per: dict[str, dict[str, float]] = {c: {} for c in self.conditions}
        for (iid, cond), records in self.slots.items():
            slots = [r.slot for r in records if r.valid]
            distances = [
                pair_distance(vectors[(iid, cond, i)], vectors[(iid, cond, j)])
                for i, j in itertools.combinations(slots, 2)
            ]
            if distances:
                per[cond][iid] = sum(distances) / len(distances)
        out: dict[str, Any] = {"model": MODEL, "revision": REVISION}
        out |= {c: ci_dict(bootstrap_mean_ci(list(v.values()))) for c, v in per.items()}
        for a, b in self.comparisons():
            common = set(per[a]) & set(per[b])
            out[f"{a}-{b}"] = ci_dict(
                paired_delta_ci({i: per[a][i] for i in common}, {i: per[b][i] for i in common})
            )
        return out

    async def run(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), "utf-8")
        try:
            await self.check_models()
            instances = self.instances()
            gen = self.cfg["generator"]
            await self.generate(instances)
            await self.generate_anchors(
                self.new_client(gen["provider"], gen["concurrency"]), instances
            )
            print("generation done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)

            judge = self.judge_client("spark")
            by_id = {x.id: x for x in instances}
            jobs = []
            for (iid, cond, slot), idea in self.valid_ideas():
                jobs.append(self.quality(judge, "anchor", by_id[iid], cond, slot, idea))
                jobs.append(self.grounding_one(judge, by_id[iid], cond, slot, idea))
            await asyncio.gather(*jobs)
            print("judges done:", dict(self.progress), f"US${self.spent_usd:.3f}", flush=True)

            invalid = {
                c: sum(not r.valid for (_, cc), recs in self.slots.items() if cc == c for r in recs)
                for c in self.conditions
            }
            result = {
                "run_id": self.run_id,
                "spent_usd": round(self.spent_usd, 4),
                "anchors_valid": sum(a is not None for a in self.anchors.values()),
                "invalid_ideas": invalid,
                "content_filtered": self.content_filtered,
                "retried_generations": sum(
                    r.attempt > 0 for recs in self.slots.values() for r in recs
                ),
                "length": self.length_report(),
                "length_vs_q": self.length_vs_q(),
                "q_anchor": self.q_report("anchor"),
                "controlled_for_length": self.controlled(self.cfg.get("n_boot", 10_000)),
                "grounding": self.grounding_report(),
                "embedding_distance": self.embedding_distance(),
                "judge_client": self.client_stats(judge),
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
        raise SystemExit("usage: python -m oblique.length_check configs/R-*.yaml")
    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    asyncio.run(LengthCheck(cfg).run())


if __name__ == "__main__":
    main(sys.argv[1:])
