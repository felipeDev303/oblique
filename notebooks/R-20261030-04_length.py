"""Local analysis of R-20261030-04: is q against the anchor confounded with idea length?

    uv run python notebooks/R-20261030-04_length.py

No API calls: reads the JSONL outputs of R-20261030-03 and -04. Also recomputes CU with
q against the anchor and the LLM distances of R-20261030-03.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from oblique.generate.schema import IdeaCard
from oblique.metrics import bootstrap_mean_ci, creative_utility, paired_delta_ci
from oblique.metrics.calibration import pearson, spearman
from oblique.prompts import q_from_orders, render_card
from oblique.seeds import seed_sequence


def rows(run: str, name: str) -> list[dict]:
    lines = Path(f"runs/{run}/{name}.jsonl").read_text("utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


anchor_len = {
    r["instance"]: len(render_card(IdeaCard(**r["idea"]))) for r in rows("R-20261030-04", "anchors")
}
ideas = {
    (r["instance"], r["condition"], r["slot"]): len(render_card(IdeaCard(**r["idea"])))
    for r in rows("R-20261030-03", "generations")
}
winners: dict[tuple, dict[bool, str]] = defaultdict(dict)
for r in rows("R-20261030-04", "judgments"):
    winners[(r["instance"], r["condition"], r["slot"])][r["idea_first"]] = r["parsed"]
q = {k: q_from_orders(v[True], v[False]) for k, v in winners.items()}

print(f"anchor chars: mean {np.mean(list(anchor_len.values())):.0f}")
for cond in ("C0", "C1"):
    keys = [k for k in q if k[1] == cond]
    length = [ideas[k] for k in keys]
    rel = [ideas[k] - anchor_len[k[0]] for k in keys]
    qs = [q[k] for k in keys]
    print(
        f"{cond}: idea chars {np.mean(length):.0f}, idea - anchor {np.mean(rel):+.0f}; "
        f"within-condition r(len, q) {pearson(length, qs):.2f}, "
        f"r(len - anchor, q) {pearson(rel, qs):.2f}, rho {spearman(rel, qs):.2f}"
    )

# OLS on ideas, CI by resampling instances: does the C1 effect survive controlling for length?
keys = sorted(q)
inst = sorted({k[0] for k in keys})
y = np.array([q[k] for k in keys])
c1 = np.array([k[1] == "C1" for k in keys], dtype=float)
rel = np.array([(ideas[k] - anchor_len[k[0]]) / 100 for k in keys])
members = {i: [j for j, k in enumerate(keys) if k[0] == i] for i in inst}


def fit(idx: list[int]) -> tuple[float, float, float]:
    plain = np.column_stack([np.ones(len(idx)), c1[idx]])
    full = np.column_stack([plain, rel[idx]])
    a = np.linalg.lstsq(plain, y[idx], rcond=None)[0]
    b = np.linalg.lstsq(full, y[idx], rcond=None)[0]
    return a[1], b[1], b[2]


gen = np.random.default_rng(seed_sequence("bootstrap"))
draws = np.array(
    [fit([j for i in gen.choice(inst, size=len(inst)) for j in members[i]]) for _ in range(10_000)]
)
point = fit(list(range(len(keys))))
for name, p, d in zip(
    ("C1 effect, no control", "C1 effect, controlling for length", "per +100 chars vs anchor"),
    point,
    draws.T,
    strict=True,
):
    lo, hi = np.quantile(d, [0.025, 0.975])
    print(f"{name}: {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")

# CU with q against the anchor and the LLM distances of the baseline.
D = {}
for r in rows("R-20261030-03", "judgments"):
    if r["judge"] == "distance-spark":
        s = r["parsed"]
        d = sum((5 - s[f]) / 4 for f in ("problem_gap", "core_claim", "method")) / 3
        D[(r["instance"], r["condition"], r["slot"], r["other_slot"])] = d
cu: dict[str, dict[str, float]] = {"C0": {}, "C1": {}}
for i in inst:
    for cond in cu:
        m = [[0.0] * 3 for _ in range(3)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            m[a][b] = m[b][a] = D[(i, cond, a, b)]
        cu[cond][i] = creative_utility([q[(i, cond, s)] for s in range(3)], m)
for cond, values in cu.items():
    ci = bootstrap_mean_ci(list(values.values()))
    print(f"CU with anchor q, {cond}: {ci.mean:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]")
ci = paired_delta_ci(cu["C1"], cu["C0"])
print(f"CU with anchor q, C1 - C0: {ci.mean:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]")
