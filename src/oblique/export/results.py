"""The data of the results page, from stored metrics only: no model call, no split B record.

    uv run python -m oblique.export.results runs/R-20261103-01 runs/R-20261031-01

Reads `metrics.json` and `post_b.json` of the confirmation run and `metrics.json` of the map, and
writes `web/results/data.js` (`window.RESULTS = {...}`), so the page also opens from disk.

- C1 against C1s on split B: CU, controlled q and d per condition with their CIs, and the paired
  differences of ADR-011.
- The map on split A: per card, controlled Δq and Δd against C2 with their CIs and the ADR-003 band;
  C2-T as one more point, and C2 at the origin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OUT = Path("web/results/data.js")


def _cell(c: dict[str, Any] | None) -> dict[str, float] | None:
    if c is None:
        return None
    return {k: round(c[k], 4) for k in ("mean", "lo", "hi")} | {"n": c["n"]}


def build(
    confirm: dict[str, Any], post_b: dict[str, Any], mapped: dict[str, Any]
) -> dict[str, Any]:
    conditions = confirm["conditions"]
    c1 = {
        name: {m: _cell(conditions[name][m]) for m in ("cu", "q_controlled", "d")}
        | {"sets": conditions[name]["sets_complete"]}
        for name in ("C1", "C1s")
    }
    cards = []
    for c in mapped["conditions"]:
        if c["condition"] == "C2":
            continue
        cards.append(
            {
                "id": c["condition"],
                "text": c["text"],
                "origin": c["origin"],
                "family": c["family"],
                "band": c["band"],
                "confirmed": c["confirmed"],
                "dq": _cell(c["q_controlled_minus_c2"]),
                "dd": _cell(c["d_minus_c2"]),
            }
        )
    reference = mapped["reference"]
    return {
        "c1_vs_c1s": {
            "run": confirm["run_id"],
            "conditions": c1,
            "delta": {k: _cell(v) for k, v in post_b["c1s_vs_c1"].items()},
            "n_boot": post_b["n_boot"],
        },
        "map": {
            "run": mapped["run_id"],
            "delta": reference["delta"],
            "c2": {
                "q_controlled": _cell(reference["q_controlled_c2"]),
                "d": _cell(reference["d_c2"]),
            },
            "ml": mapped["ml"]["cards"],
            "cards": cards,
        },
    }


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("usage: python -m oblique.export.results <confirm run dir> <map run dir>")
    confirm_dir, map_dir = (Path(a) for a in argv)
    data = build(
        json.loads((confirm_dir / "metrics.json").read_text("utf-8")),
        json.loads((confirm_dir / "post_b.json").read_text("utf-8")),
        json.loads((map_dir / "metrics.json").read_text("utf-8")),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = "window.RESULTS = " + json.dumps(data, indent=1, ensure_ascii=False) + ";\n"
    OUT.write_text(text, "utf-8", newline="\n")
    print(f"{OUT}: {len(data['map']['cards'])} map points")


if __name__ == "__main__":
    main(sys.argv[1:])
