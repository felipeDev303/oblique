"""A real trial of "bring your papers", through the app's own API.

    uv run --env-file ../oblique-repo/.env python -m oblique.product.trial configs/R-*.yaml

1. Uploads an accepted paper with one past the page limit: it must be refused with no call.
2. Generates a set from the two accepted papers, and reports the real cost of each call.
3. Generates a second set from the same papers: the sheets come from the cache.

The PDFs go in `pdf_dir`, which is not committed. The summary keeps IDs and licenses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi.testclient import TestClient

from oblique.product.app import create_app

POLL_SECONDS = 2.0
SET_TIMEOUT_SECONDS = 1200.0


def fetch(paper: dict[str, Any], directory: Path) -> tuple[str, bytes]:
    name = f"{paper['arxiv_id']}{paper['version']}.pdf"
    path = directory / name
    if not path.exists():
        url = f"https://arxiv.org/pdf/{paper['arxiv_id']}{paper['version']}"
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return name, path.read_bytes()


def ledger_rows(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.exists() else []
    )


def upload(client: TestClient, files: list[tuple[str, bytes]]) -> httpx.Response:
    return client.post("/api/sets", files=[("files", (n, d, "application/pdf")) for n, d in files])


def wait_for_set(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + SET_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        body = client.get(f"/api/sets/{job_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"set {job_id} did not finish")


def one_set(client: TestClient, ledger: Path, files: list[tuple[str, bytes]]) -> dict[str, Any]:
    before = len(ledger_rows(ledger))
    started = time.monotonic()
    response = upload(client, files)
    if response.status_code != 202:
        return {"http_status": response.status_code, "detail": response.json()["detail"]}
    body = wait_for_set(client, response.json()["id"])
    calls = ledger_rows(ledger)[before:]
    return {
        "http_status": response.status_code,
        "status": body["status"],
        "message": body["message"],
        "seconds": round(time.monotonic() - started, 1),
        "paid_calls": calls,
        "cost_usd": sum(c["cost_usd"] for c in calls),
        "sheets": body["sheets"],
        "ideas": body["ideas"],
    }


def render_summary(cfg: dict[str, Any], result: dict[str, Any]) -> str:
    lines = [f"# {cfg['run_id']} · Trial of bring your papers", "", "## Papers", ""]
    lines += ["| arXiv | License | Pages | Role | Title |", "|---|---|---|---|---|"]
    for p in cfg["papers"]:
        lines.append(
            f"| {p['arxiv_id']}{p['version']} | {p['license']} | {p['pages']} | {p['role']} "
            f"| {p['title']} |"
        )
    refused = result["refused"]
    lines += [
        "",
        "PDFs are not redistributed here; each one is available from arXiv under its license.",
        "",
        "## The long PDF",
        "",
        f"HTTP {refused['http_status']}: {refused['detail']} "
        f"Paid calls: {refused['paid_calls']}, cost US${refused['cost_usd']:.4f}.",
        "",
        "## Cost per run",
        "",
        "| Run | Status | Seconds | Sheet calls | Idea calls | In | Cached | Out | US$ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in (("first", result["first"]), ("second", result["second"])):
        calls = run.get("paid_calls", [])
        lines.append(
            f"| {name} | {run.get('status', run.get('detail'))} | {run.get('seconds', '')} "
            f"| {sum(c['kind'] == 'sheet' for c in calls)} "
            f"| {sum(c['kind'] == 'idea' for c in calls)} "
            f"| {sum(c['tokens_in'] for c in calls)} | {sum(c['tokens_cached'] for c in calls)} "
            f"| {sum(c['tokens_out'] for c in calls)} | {run.get('cost_usd', 0):.4f} |"
        )
    lines += ["", f"Total: US${result['total_usd']:.4f}.", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.product.trial configs/R-*.yaml")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    directory = Path(cfg["pdf_dir"])
    papers = {p["role"]: [] for p in cfg["papers"]}
    for p in cfg["papers"]:
        papers[p["role"]].append(fetch(p, directory))
    accepted, long_paper = papers["accepted"], papers["long"]
    out = Path("runs") / cfg["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    ledger = Path(cfg["ledger"])
    result: dict[str, Any] = {}
    with TestClient(create_app(cfg)) as client:
        before = len(ledger_rows(ledger))
        response = upload(client, [accepted[0], long_paper[0]])
        calls = ledger_rows(ledger)[before:]
        result["refused"] = {
            "http_status": response.status_code,
            "detail": response.json().get("detail"),
            "paid_calls": len(calls),
            "cost_usd": sum(c["cost_usd"] for c in calls),
        }
        print(json.dumps(result["refused"], ensure_ascii=False), flush=True)
        result["first"] = one_set(client, ledger, accepted)
        print("first:", result["first"].get("status"), result["first"].get("cost_usd"), flush=True)
        result["second"] = one_set(client, ledger, accepted)
        print(
            "second:", result["second"].get("status"), result["second"].get("cost_usd"), flush=True
        )
    result["total_usd"] = sum(c["cost_usd"] for c in ledger_rows(ledger))
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    (out / "summary.md").write_text(render_summary(cfg, result), "utf-8", newline="\n")
    print(f"total US${result['total_usd']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
