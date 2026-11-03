"""The two example papers: their sheets and the one-call set, computed in advance.

    uv run python -m oblique.product.examples runs/R-20261102-04 configs/R-20261102-04.yaml \
        examples/papers/example.json

"Try with 2 example papers" starts a set without an upload. The sheets of the two CC BY papers and
the one-call column (K=3, no card) come from this file; only the Oblique sequential set is
generated live. The PDFs themselves are not packaged: the page links to them on arXiv.

The file is built from a real trial (`sets.json` of R-20261102-04) and records the prompt hashes it
was made with. The app refuses a file whose hashes differ from the current prompts, since its
sheets or its one-call set would no longer be what those prompts give.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from oblique.product.documents import Evidence, Sheet
from oblique.prompts.product import DOCUMENT_SHEET_V1, IDEATION_SHEETS_ONE_CALL_V1


def current_prompts() -> dict[str, str]:
    return {"sheet": DOCUMENT_SHEET_V1.hash, "one_call": IDEATION_SHEETS_ONE_CALL_V1.hash}


def sheet_from_public(d: dict[str, Any]) -> Sheet:
    return Sheet(
        number=d["document"],
        name=d["name"],
        title=d["title"],
        summary=d["summary"],
        methods=d["methods"],
        limitations=d["limitations"],
        evidence=tuple(Evidence(e["id"], e["quote"], e["found"]) for e in d["evidence"]),
    )


@dataclass(frozen=True)
class Example:
    papers: list[dict[str, Any]]
    sheets: list[Sheet]
    one_call: list[dict[str, Any]]
    documents: list[dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "precomputed": "The sheets and the one-call column were generated in advance; "
            "the Oblique sequential set is generated live.",
        }


def load_example(path: str | Path) -> Example:
    data = json.loads(Path(path).read_text("utf-8"))
    if data["prompts"] != current_prompts():
        raise SystemExit(f"{path} was built with other prompts: rebuild it from a new trial")
    papers = data["papers"]
    if any(not p["license"].startswith("CC BY") for p in papers):
        raise SystemExit("every example paper must be CC BY")
    return Example(
        papers=papers,
        sheets=[sheet_from_public(s) for s in data["sheets"]],
        one_call=data["one_call"],
        documents=[{"number": p["number"], "name": p["name"], "pages": p["pages"]} for p in papers],
    )


def build(page: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """The example file from the page body of a side-by-side trial set and the trial's config."""
    if page["view"] != "side_by_side" or page["status"] != "done":
        raise SystemExit("the trial set must be a finished side-by-side set")
    by_name = {f"{p['arxiv_id']}{p['version']}.pdf": p for p in cfg["papers"]}
    papers = []
    for d in page["documents"]:
        p = by_name[d["name"]]
        papers.append(
            {
                "number": d["number"],
                "name": d["name"],
                "pages": d["pages"],
                "arxiv_id": p["arxiv_id"],
                "version": p["version"],
                "title": p["title"],
                "authors": p["authors"],
                "license": p["license"],
                "url": f"https://arxiv.org/abs/{p['arxiv_id']}{p['version']}",
            }
        )
    return {
        "source_run": cfg["run_id"],
        "model": cfg["generator"]["model"],
        "prompts": current_prompts(),
        "papers": papers,
        "sheets": page["sheets"],
        "one_call": page["one_call"],
    }


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit("usage: python -m oblique.product.examples <trial run dir> <config> <out>")
    run, config, out = (Path(a) for a in argv)
    page = json.loads((run / "sets.json").read_text("utf-8"))[0]
    cfg = yaml.safe_load(config.read_text("utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(build(page, cfg), indent=1, ensure_ascii=False) + "\n"
    out.write_text(text, "utf-8", newline="\n")
    load_example(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
