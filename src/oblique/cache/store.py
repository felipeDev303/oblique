"""Local response cache backed by SQLite (WAL).

Every model call goes through this cache. Entries are never deleted: the
database holds every paid response, so back it up outside the repo.

Only successful API responses are stored. A response whose text does not
parse as JSON is still a successful response and is cached; validation
happens later (ADR-006).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key              TEXT PRIMARY KEY,
    model            TEXT NOT NULL,
    request          TEXT NOT NULL,
    response         TEXT NOT NULL,
    tokens_in        INTEGER NOT NULL DEFAULT 0,
    tokens_cached    INTEGER NOT NULL DEFAULT 0,
    tokens_out       INTEGER NOT NULL DEFAULT 0,
    tokens_reasoning INTEGER NOT NULL DEFAULT 0,
    created          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


def make_key(
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    sample_idx: int = 0,
    attempt: int = 0,
    *,
    provider: str,
) -> str:
    """Hash of everything that determines a response.

    `provider` separates the same model and request served by different
    endpoints or accounts (Model API, Baseten), which may answer differently.
    `sample_idx` separates independent samples of the same request (e.g. the
    three draws of C0), which would otherwise collide in the cache.
    `attempt` separates a retry from the original call, so a retry never
    returns the cached failure and never collides with another sample.
    """
    blob = json.dumps(
        {"pr": provider, "m": model, "msg": messages, "p": params, "s": sample_idx, "a": attempt},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """Thread-safe key -> response store. One writer at a time; reads never block it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute(_SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()

    key = staticmethod(make_key)

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        key: str,
        *,
        model: str,
        request: dict[str, Any],
        response: dict[str, Any],
        tokens_in: int = 0,
        tokens_cached: int = 0,
        tokens_out: int = 0,
        tokens_reasoning: int = 0,
    ) -> None:
        """Store a response. An existing key is never overwritten."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO cache (key, model, request, response, tokens_in,"
                " tokens_cached, tokens_out, tokens_reasoning) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    model,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    tokens_in,
                    tokens_cached,
                    tokens_out,
                    tokens_reasoning,
                ),
            )

    def __contains__(self, key: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM cache WHERE key = ?", (key,)).fetchone()
        return row is not None

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
