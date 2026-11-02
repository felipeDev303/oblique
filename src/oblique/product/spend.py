"""The spend of the app and the proxy: one ledger file, one budget.

Both append to the same file and read what the other wrote, so the budget holds across processes
and restarts. Limits are checked before each call, so a limit can be passed by the call in flight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oblique.llm.client import Usage


def call_cost(usage: Usage, price: dict[str, float]) -> float:
    """USD at prices per million tokens. Reasoning tokens are already inside `tokens_out`."""
    uncached = usage.tokens_in - usage.tokens_cached
    return (
        uncached * price["in"]
        + usage.tokens_cached * price["cached"]
        + usage.tokens_out * price["out"]
    ) / 1e6


class SpendLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._read_bytes = 0
        self._total = 0.0

    def total(self) -> float:
        """USD so far, including what another process has written since the last read."""
        if not self.path.exists():
            return self._total
        with self.path.open("rb") as f:
            f.seek(self._read_bytes)
            chunk = f.read()
        complete = chunk[: chunk.rfind(b"\n") + 1]
        self._read_bytes += len(complete)
        for line in complete.splitlines():
            self._total += json.loads(line)["cost_usd"]
        return self._total

    def record(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
