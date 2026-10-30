"""GiantsBench-test instances and the A/B splits (ADR-007).

Source: `giants2026/GiantsBench-test` on Hugging Face (CC BY 4.0), pinned to
one revision and checked by SHA256. The file is not redistributed; `fetch()`
downloads it into `data/`.

An instance is one row: two parent-paper summaries (the corpus) and the gold
insight, taken from the single `<insight>...</insight>` block of `completion`.
Instance IDs are `i-NNNN`, the row index in the pinned file; split files also
store `pair_id` so a different file can never be read silently.

Splits. The dataset has no date field, but every `arxiv_id` is new-style
(YYMM.NNNNN), assigned at submission, so (YYMM, number) orders the downstream
papers in time. The 100 most recent instances are taken per domain with equal
allocation (100 // 16 = 6 each; the remaining 4 go to domains drawn with the
`splits` seed), then each domain is split half and half between A and B.
Domains with an odd count give their extra instance to A or B, balanced so
both splits hold 50. Rows whose `arxiv_id` appears more than once are excluded.

Split B is sealed: `load_instances("B")` raises unless OBLIQUE_UNSEAL_B=1.
Split IDs are not instance content and can be listed without unsealing.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from oblique.seeds import rng

DATASET = "giants2026/GiantsBench-test"
REVISION = "6095cb1386e4d12a09c981b2448c28b4c62381a1"
LICENSE = "CC BY 4.0"
_REMOTE_FILE = "data/train-00000-of-00001.parquet"
SHA256 = "ce1d7668717cd3e54e3862346a116aad622d5eeefa06bebdba90cfee4c705beb"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "giantsbench" / f"test-{REVISION[:7]}.parquet"
SPLITS_DIR = REPO_ROOT / "splits"

SPLITS = ("A", "B")
PER_SPLIT = 50
UNSEAL_ENV = "OBLIQUE_UNSEAL_B"

_ARXIV = re.compile(r"(\d{4})\.(\d{4,5})(?:v\d+)?")
_INSIGHT = re.compile(r"<insight>(.*?)</insight>", re.DOTALL)
_ID = re.compile(r"i-(\d{4,})")


class SealedSplitError(PermissionError):
    pass


class Instance(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    pair_id: str
    arxiv_id: str
    domain: str
    paper1_title: str
    paper2_title: str
    abstract_1: str
    abstract_2: str
    gold: str


def instance_id(row: int) -> str:
    return f"i-{row:04d}"


def row_index(iid: str) -> int:
    match = _ID.fullmatch(iid)
    if match is None:
        raise ValueError(f"not an instance id: {iid!r}")
    return int(match.group(1))


def recency_key(arxiv_id: str) -> tuple[int, int]:
    """(YYMM, number) of a new-style arXiv ID; larger is more recent."""
    match = _ARXIV.fullmatch(arxiv_id)
    if match is None:
        raise ValueError(f"not a new-style arXiv id: {arxiv_id!r}")
    return int(match.group(1)), int(match.group(2))


def extract_insight(completion: str) -> str:
    found = _INSIGHT.findall(completion)
    if len(found) != 1:
        raise ValueError(f"expected one <insight> block, found {len(found)}")
    return found[0].strip()


# --- data file ---------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(path: Path = DATA_PATH) -> Path:
    """Download the pinned file (once) and verify its SHA256."""
    import httpx

    if not path.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{_REMOTE_FILE}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        tmp.replace(path)
    if sha256(path) != SHA256:
        raise RuntimeError(f"{path}: SHA256 mismatch; expected {DATASET}@{REVISION[:7]}")
    return path


def read_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m oblique.bench fetch`")
    return pq.read_table(path).to_pylist()


# --- splits ------------------------------------------------------------------


def build_splits(
    rows: Sequence[dict[str, Any]], per_split: int = PER_SPLIT
) -> dict[str, list[tuple[str, str]]]:
    """Most recent 2 * per_split rows, equal allocation per domain, halved into A and B.

    Returns {"A": [(instance_id, pair_id), ...], "B": [...]}, each sorted by ID.
    """
    arxiv_counts = Counter(r["arxiv_id"] for r in rows)
    by_domain: dict[str, list[tuple[tuple[int, int], str, int]]] = defaultdict(list)
    for i, r in enumerate(rows):
        if arxiv_counts[r["arxiv_id"]] == 1:
            by_domain[r["domain"]].append((recency_key(r["arxiv_id"]), r["pair_id"], i))

    domains = sorted(by_domain)
    total = 2 * per_split
    base, extra = divmod(total, len(domains))
    gen = rng("splits")
    extra_domains = set(gen.choice(domains, size=extra, replace=False).tolist())
    quota = {d: base + (d in extra_domains) for d in domains}

    selected: dict[str, list[int]] = {}
    for d in domains:
        newest = sorted(by_domain[d], reverse=True)
        if len(newest) < quota[d]:
            raise ValueError(f"domain {d!r} has {len(newest)} eligible rows, needs {quota[d]}")
        selected[d] = [i for _, _, i in newest[: quota[d]]]

    # Odd quotas leave one swing instance per domain; half of them go to A.
    odd = [d for d in domains if quota[d] % 2]
    swing_to_a = {odd[j] for j in gen.permutation(len(odd))[: len(odd) // 2]}

    out: dict[str, list[tuple[str, str]]] = {"A": [], "B": []}
    for d in domains:
        n_a = quota[d] // 2 + (d in swing_to_a)
        order = [selected[d][j] for j in gen.permutation(len(selected[d]))]
        for split, idx in (("A", order[:n_a]), ("B", order[n_a:])):
            out[split] += [(instance_id(i), rows[i]["pair_id"]) for i in idx]
    for split in out:
        out[split].sort()
    return out


def write_splits(splits: dict[str, list[tuple[str, str]]], directory: Path = SPLITS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, entries in splits.items():
        header = (
            f"# {DATASET}@{REVISION[:7]} ({LICENSE}) - split {name}, {len(entries)} instances\n"
            "# built by `python -m oblique.bench splits` (seed purpose 'splits')\n"
            "# instance_id<TAB>pair_id\n"
        )
        lines = "".join(f"{iid}\t{pid}\n" for iid, pid in entries)
        (directory / f"{name}.txt").write_text(header + lines, encoding="utf-8", newline="\n")


def split_ids(name: str, directory: Path = SPLITS_DIR) -> list[tuple[str, str]]:
    """(instance_id, pair_id) pairs of a split. IDs only: not sealed."""
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; expected one of {SPLITS}")
    text = (directory / f"{name}.txt").read_text(encoding="utf-8")
    entries = []
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            iid, pid = line.split("\t")
            entries.append((iid, pid))
    return entries


def load_instances(
    name: str, *, path: Path = DATA_PATH, directory: Path = SPLITS_DIR
) -> list[Instance]:
    if name == "B" and os.environ.get(UNSEAL_ENV) != "1":
        raise SealedSplitError(
            f"split B is sealed; set {UNSEAL_ENV}=1 only when the author says so"
        )
    entries = split_ids(name, directory)
    rows = read_rows(path)
    instances = []
    for iid, pid in entries:
        r = rows[row_index(iid)]
        if r["pair_id"] != pid:
            raise ValueError(f"{iid}: pair_id {r['pair_id']!r} != {pid!r}; wrong data file?")
        instances.append(
            Instance(
                id=iid,
                pair_id=pid,
                arxiv_id=r["arxiv_id"],
                domain=r["domain"],
                paper1_title=r["paper1_title"],
                paper2_title=r["paper2_title"],
                abstract_1=r["abstract_1"],
                abstract_2=r["abstract_2"],
                gold=extract_insight(r["completion"]),
            )
        )
    return instances


def main(argv: Sequence[str]) -> None:
    cmd = argv[0] if argv else ""
    if cmd == "fetch":
        print(fetch())
    elif cmd == "splits":
        rows = read_rows(fetch())
        splits = build_splits(rows)
        write_splits(splits)
        # Counts only: nothing from split B is printed.
        for name, entries in splits.items():
            print(name, len(entries), "instances ->", SPLITS_DIR / f"{name}.txt")
    else:
        raise SystemExit("usage: python -m oblique.bench fetch|splits")


if __name__ == "__main__":
    main(sys.argv[1:])
