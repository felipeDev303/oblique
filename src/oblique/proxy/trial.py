"""A real trial of the proxy: the running server, through the OpenAI SDK.

    uv run --env-file <path to .env> python -m oblique.proxy.trial configs/R-*.yaml

Starts `oblique.proxy.app` as its own process, checks that it listens only on localhost, and sends
one request without opt-in and one with it. The ledger comes from the trial's base config.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import openai
import yaml

from oblique.product.spend import SpendLedger

START_TIMEOUT_SECONDS = 60


def wait_until_up(url: str) -> None:
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            httpx.post(url, json={}, timeout=2)
            return
        except httpx.TransportError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not come up")


def listening_addresses(port: int) -> list[str]:
    """The local addresses listening on `port`, from the operating system."""
    out = subprocess.run(["netstat", "-an"], capture_output=True, text=True, check=True).stdout
    return parse_listening(out, port)


def parse_listening(netstat: str, port: int) -> list[str]:
    """A listening socket has no remote address. The word for its state depends on the language of
    the system, so it is not read."""
    addresses = []
    for line in netstat.splitlines():
        fields = line.split()
        if (
            len(fields) >= 4
            and fields[0].upper() == "TCP"
            and fields[1].endswith(f":{port}")
            and fields[2] in ("0.0.0.0:0", "[::]:0", "*:*")
        ):
            addresses.append(fields[1])
    return addresses


def render_summary(cfg: dict[str, Any], result: dict[str, Any]) -> str:
    plain, oblique = result["passthrough"], result["oblique"]
    lines = [
        f"# {cfg['run_id']} · Real trial of the proxy",
        "",
        f"Listening on: {', '.join(result['listening'])}. Total: US${result['total_usd']:.4f}.",
        "",
        "| Request | Calls | Tokens in | Tokens out | US$ | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in (("muse-spark-1.3", plain), ("muse-spark-1.3+oblique", oblique)):
        lines.append(
            f"| {name} | {r['calls']} | {r['usage']['prompt_tokens']} "
            f"| {r['usage']['completion_tokens']} | {r['usd']:.4f} | {r['seconds']} |"
        )
    cards = ", ".join(c["id"] for c in oblique["cards_used"])
    lines += ["", f"Cards used: {cards}.", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.proxy.trial configs/R-*.yaml")
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(cfg["product_config"]).read_text(encoding="utf-8"))
    ledger = SpendLedger(base["ledger"])
    out = Path("runs") / cfg["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{cfg['port']}/v1"
    server = subprocess.Popen(
        [sys.executable, "-m", "oblique.proxy.app", argv[0]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result: dict[str, Any] = {}
    try:
        wait_until_up(f"{url}/chat/completions")
        result["listening"] = listening_addresses(cfg["port"])
        client = openai.OpenAI(base_url=url, api_key="local-client", max_retries=0, timeout=600)
        messages = [{"role": "user", "content": cfg["question"]}]
        for name, model in (
            ("passthrough", "muse-spark-1.3"),
            ("oblique", "muse-spark-1.3+oblique"),
        ):
            before = ledger.total()
            lines = len(ledger.path.read_text("utf-8").splitlines()) if ledger.path.exists() else 0
            started = time.monotonic()
            answer = client.chat.completions.create(model=model, messages=messages)
            data = answer.model_dump()
            after = len(ledger.path.read_text("utf-8").splitlines())
            result[name] = {
                "seconds": round(time.monotonic() - started, 1),
                "calls": after - lines,
                "usd": ledger.total() - before,
                "model": data["model"],
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                },
                "finish_reason": data["choices"][0]["finish_reason"],
                "content": data["choices"][0]["message"]["content"],
                "cards_used": (data.get("oblique") or {}).get("cards_used"),
                "answers": (data.get("oblique") or {}).get("answers"),
                "stopped": (data.get("oblique") or {}).get("stopped"),
            }
            print(name, result[name]["calls"], "calls", f"US${result[name]['usd']:.4f}", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=10)
    result["total_usd"] = ledger.total()
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    (out / "summary.md").write_text(render_summary(cfg, result), "utf-8", newline="\n")
    print(f"total US${result['total_usd']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
