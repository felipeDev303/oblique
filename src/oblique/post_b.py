"""Analyses after split B, from the stored records of the run only (ADR-011). All exploratory.

    uv run --group embed python -m oblique.post_b configs/R-20261103-01.yaml

No model is called: the generations, anchors and judgments come from `runs/<run_id>/*.jsonl`, and
the embedding distance is recomputed locally. Split B itself is not opened. Before anything new is
computed, the rebuilt sets and the controlled q of E6 - C1s must equal those of `metrics.json`.

- controlled q, MG - C1s, which the product rule needs (ADR-010, amendment 3);
- C1s - C1 in CU, controlled q, q against the gold and d, paired, 2,000 resamples;
- d within the set of R, E6, ML and MG against C1s;
- the product deck: among ML, MG and E6, the highest ΔCU against C1s among those whose controlled
  q has a lower bound above -δ and whose mean Δgrounding is above the guard (ADR-011: the guard
  applies to the mean). If none passes, the highest controlled Δq, marked experimental.

The section is appended to `summary.md` as "Post-B exploratory (ADR-011)", replacing an earlier one.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from oblique.confirm import BASELINE, ConfirmB, _ci, _ci_n
from oblique.generate.records import GenerationRecord
from oblique.metrics.calibration import controlled_win_rates
from oblique.prompts import distance, grounding, q_from_orders

PRODUCT_CANDIDATES = ("ML", "MG", "E6")
WITH_CARDS = ("R", "E6", "ML", "MG")
SECTION = "## Post-B exploratory (ADR-011)"
QUALITY = {"quality-anchor": "anchor", "quality-spark": "spark", "quality-wonder": "wonder"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def final_values(rows: list[dict[str, Any]]) -> dict[tuple, Any]:
    """The value each judge call ended with: its last attempt, or None if that attempt failed to
    parse or was filtered (ADR-006, amendment 6)."""
    last: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if r["judge"].startswith("quality"):
            key = (r["judge"], r["instance"], r["condition"], r["slot"], r["idea_first"])
        elif r["judge"] == "distance-spark":
            key = (r["judge"], r["instance"], r["condition"], r["slot"], r["other_slot"])
        else:
            key = (r["judge"], r["instance"], r["condition"], r["slot"])
        if key not in last or r["attempt"] >= last[key]["attempt"]:
            last[key] = r
    return {k: (r["parsed"] if r["parse_error"] is None else None) for k, r in last.items()}


def product_deck(
    q_controlled: dict[str, dict[str, Any]],
    cu: dict[str, dict[str, Any]],
    grounding_delta: dict[str, dict[str, Any]],
    delta: float,
    guard: float,
) -> dict[str, Any]:
    """ADR-010, amendment 3, with the guard on the mean (ADR-011)."""
    checks = {
        c: {
            "delta_cu": cu[c]["mean"],
            "q_controlled": q_controlled[c]["mean"],
            "q_controlled_lo": q_controlled[c]["lo"],
            "non_inferior": q_controlled[c]["lo"] > -delta,
            "grounding_mean": grounding_delta[c]["mean"],
            "guard": grounding_delta[c]["mean"] > guard,
        }
        for c in PRODUCT_CANDIDATES
    }
    passing = [c for c, v in checks.items() if v["non_inferior"] and v["guard"]]
    if passing:
        deck, experimental = max(passing, key=lambda c: checks[c]["delta_cu"]), False
    else:
        deck = max(PRODUCT_CANDIDATES, key=lambda c: q_controlled[c]["mean"])
        experimental = True
    return {"deck": deck, "experimental": experimental, "passing": passing, "checks": checks}


class PostB(ConfirmB):
    def load(self) -> None:
        """The state the run had before its report, from its records."""
        records = [GenerationRecord(**r) for r in read_jsonl(self.out / "generations.jsonl")]
        slots: dict[tuple[str, str], list[GenerationRecord]] = defaultdict(list)
        for r in records:
            slots[(r.instance, r.condition)].append(r)
        self.slots = {k: sorted(v, key=lambda r: r.slot) for k, v in sorted(slots.items())}
        anchors = [GenerationRecord(**r) for r in read_jsonl(self.out / "anchors.jsonl")]
        self.anchors = {r.instance: r.idea for r in anchors}

        rows = read_jsonl(self.out / "judgments.jsonl")
        self.content_filtered = [r for r in rows if r["parse_error"] == "content_filter"]
        values = final_values(rows)
        for key, value in values.items():
            judge = key[0]
            if judge in QUALITY:
                continue
            if judge == "distance-spark":
                _, iid, cond, i, j = key
                self.d_llm[(iid, cond, i, j)] = None if value is None else distance(value)
            elif judge == "grounding-spark":
                _, iid, cond, slot = key
                self.ground[(iid, cond, slot)] = None if value is None else grounding(value)
        for judge, name in QUALITY.items():
            ideas = {k[1:4] for k in values if k[0] == judge}
            for iid, cond, slot in ideas:
                winners = (values.get((judge, iid, cond, slot, True)),
                           values.get((judge, iid, cond, slot, False)))  # fmt: skip
                if None in winners:
                    self.q[name][(iid, cond, slot)] = None
                else:
                    self.q[name][(iid, cond, slot)] = q_from_orders(*winners)
                    self.verdicts[name][(iid, cond, slot)] = winners

    def check_against_run(self, stored: dict[str, Any], other_q: dict[str, Any]) -> None:
        """The rebuilt state must give the run's own numbers before anything new is read."""
        sets = [self.set_metrics(k) for k in sorted(self.slots)]
        if json.loads(json.dumps(sets, default=str)) != stored["sets"]:
            raise SystemExit("the rebuilt sets differ from metrics.json")
        expected = stored["exploratory"]["h6_q_controlled_e6_c1s"]
        if other_q["differences"]["E6-C1s"] != expected:
            raise SystemExit("the rebuilt regression differs from metrics.json (E6 - C1s)")

    def analyses(self) -> dict[str, Any]:
        stored = json.loads((self.out / "metrics.json").read_text("utf-8"))
        n_boot = self.cfg["n_boot_other"]
        rows = self.regression_rows()
        comparisons = [(c, BASELINE) for c in self.conditions if c != BASELINE] + [("C1s", "C1")]
        q = controlled_win_rates(rows, self.conditions, comparisons=comparisons, n_boot=n_boot)
        self.check_against_run(stored, q)

        sets = [self.set_metrics(k) for k in sorted(self.slots)]
        by = {c: {r["instance"]: r for r in sets if r["complete"] and r["condition"] == c}
              for c in self.conditions}  # fmt: skip

        def compare(a: str, b: str, metric: str) -> dict[str, Any] | None:
            return self.paired(by, metric, a, b, n_boot)

        q_vs_c1s = {c: q["differences"][f"{c}-{BASELINE}"] for c in PRODUCT_CANDIDATES}
        cu_vs_c1s = {c: compare(c, BASELINE, "cu") for c in PRODUCT_CANDIDATES}
        grounding_vs_c1s = {c: compare(c, BASELINE, "grounding") for c in PRODUCT_CANDIDATES}
        return {
            "run_id": self.run_id,
            "n_boot": n_boot,
            "rebuilt_matches_run": True,
            "q_controlled_mg_c1s": q["differences"]["MG-C1s"],
            "c1s_vs_c1": {
                "cu": compare("C1s", "C1", "cu"),
                "q_controlled": q["differences"]["C1s-C1"],
                "q_gold": compare("C1s", "C1", "q_gold"),
                "d": compare("C1s", "C1", "d"),
                "d_embed": compare("C1s", "C1", "d_embed"),
            },
            "d_vs_c1s": {
                c: {"d": compare(c, BASELINE, "d"), "d_embed": compare(c, BASELINE, "d_embed")}
                for c in WITH_CARDS
            },
            "product": product_deck(
                q_vs_c1s,
                cu_vs_c1s,
                grounding_vs_c1s,
                self.cfg["delta"],
                self.cfg["grounding_guard"],
            ),
            "product_inputs": {
                "q_controlled_vs_c1s": q_vs_c1s,
                "cu_vs_c1s": cu_vs_c1s,
                "grounding_vs_c1s": grounding_vs_c1s,
            },
        }


def render_section(result: dict[str, Any]) -> str:
    c1 = result["c1s_vs_c1"]
    product = result["product"]
    lines = [
        SECTION,
        "",
        "Written after this run was read, from its stored records, with no model call. "
        f"{result['n_boot']:,} resamples. None of it was pre-registered.",
        "",
        f"- Controlled q, MG - C1s: {_ci(result['q_controlled_mg_c1s'])}",
        "",
        "C1s - C1 (sequential set against one call), paired:",
        "",
        f"- CU: {_ci_n(c1['cu'])}",
        f"- Controlled q: {_ci(c1['q_controlled'])}",
        f"- q against the gold: {_ci_n(c1['q_gold'])}",
        f"- d: {_ci_n(c1['d'])}; with embeddings {_ci(c1['d_embed'])}",
        "",
        "d within the set, against C1s:",
        "",
        "| Condition | Δd | Δd embeddings |",
        "|---|---|---|",
        *(
            f"| {c} | {_ci_n(v['d'])} | {_ci(v['d_embed'])} |"
            for c, v in result["d_vs_c1s"].items()
        ),
        "",
        "Product deck (ADR-010, amendment 3; guard on the mean, ADR-011):",
        "",
        "| Deck | ΔCU vs C1s | Controlled Δq | Lower bound | Non-inferior | Mean Δgrounding "
        "| Guard |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {c} | {v['delta_cu']:.3f} | {v['q_controlled']:.3f} | {v['q_controlled_lo']:.3f} "
            f"| {'yes' if v['non_inferior'] else 'no'} | {v['grounding_mean']:.3f} "
            f"| {'yes' if v['guard'] else 'no'} |"
            for c, v in product["checks"].items()
        ),
        "",
        f"**Deck: {product['deck']}**"
        + (
            ", marked experimental: no deck passed both checks, so the rule takes the highest "
            "controlled Δq."
            if product["experimental"]
            else "."
        ),
        "",
        "All three decks use the same 2,000 resamples here, so the lower bound for ML differs a "
        "little from the one in H1, which uses 10,000.",
    ]
    return "\n".join(lines) + "\n"


def append_section(summary: Path, section: str) -> None:
    text = summary.read_text("utf-8")
    if SECTION in text:
        text = text[: text.index(SECTION)].rstrip("\n") + "\n"
    summary.write_text(text + "\n" + section, "utf-8", newline="\n")


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.post_b configs/R-*.yaml")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    run = PostB(cfg)
    run.load()
    run.embed()
    result = run.analyses()
    (run.out / "post_b.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), "utf-8", newline="\n"
    )
    append_section(run.out / "summary.md", render_section(result))
    print(json.dumps(result["product"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
