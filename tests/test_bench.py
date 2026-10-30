"""Benchmark loading and splits. Synthetic rows only: the real file is never needed."""

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from oblique import bench
from oblique.bench import (
    Instance,
    SealedSplitError,
    build_splits,
    extract_insight,
    instance_id,
    load_instances,
    pick_baseline,
    recency_key,
    row_index,
    split_ids,
    write_splits,
)


def _rows(n_domains: int = 16, per_domain: int = 12) -> list[dict]:
    rows = []
    for d in range(n_domains):
        for j in range(per_domain):
            arxiv = f"25{(j % 12) + 1:02d}.{10000 + 100 * d + j:05d}"
            rows.append(
                {
                    "pair_id": f"p{d:02d}_{j:02d}",
                    "arxiv_id": arxiv,
                    "domain": f"D{d:02d}",
                    "paper1_title": "t1",
                    "paper2_title": "t2",
                    "abstract_1": "a1",
                    "abstract_2": "a2",
                    "completion": f"<think>x</think><insight> gold {d}-{j} </insight>",
                }
            )
    return rows


def test_ids_round_trip():
    assert instance_id(42) == "i-0042"
    assert row_index("i-0042") == 42
    with pytest.raises(ValueError):
        row_index("0042")


def test_recency_key_orders_by_month_then_number():
    assert recency_key("2601.00001") > recency_key("2512.99999")
    assert recency_key("2510.10000v2") == (2510, 10000)
    with pytest.raises(ValueError):
        recency_key("math/0601001")


def test_extract_insight_needs_exactly_one_block():
    assert extract_insight("<think>a</think>\n<insight>\n gold \n</insight>") == "gold"
    with pytest.raises(ValueError):
        extract_insight("no insight")
    with pytest.raises(ValueError):
        extract_insight("<insight>a</insight><insight>b</insight>")


def test_splits_are_50_50_disjoint_and_stratified():
    rows = _rows()
    splits = build_splits(rows)
    a, b = ({i for i, _ in splits[s]} for s in "AB")
    assert len(a) == len(b) == 50
    assert not a & b
    for s in "AB":
        per_domain = Counter(rows[row_index(i)]["domain"] for i, _ in splits[s])
        assert len(per_domain) == 16
        assert max(per_domain.values()) - min(per_domain.values()) <= 1


def test_splits_take_the_most_recent_per_domain():
    rows = _rows()
    chosen = {row_index(i) for s in "AB" for i, _ in build_splits(rows)[s]}
    for d in {r["domain"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["domain"] == d]
        picked = [i for i in idx if i in chosen]
        left = [i for i in idx if i not in chosen]
        newest_left = max(recency_key(rows[i]["arxiv_id"]) for i in left)
        assert all(recency_key(rows[i]["arxiv_id"]) > newest_left for i in picked)


def test_splits_are_reproducible():
    rows = _rows()
    assert build_splits(rows) == build_splits(rows)


def test_duplicated_arxiv_ids_are_excluded():
    rows = _rows()
    newest = max(range(len(rows)), key=lambda i: recency_key(rows[i]["arxiv_id"]))
    rows.append({**rows[newest], "pair_id": "dup"})
    chosen = {pid for s in "AB" for _, pid in build_splits(rows)[s]}
    assert rows[newest]["pair_id"] not in chosen and "dup" not in chosen


def test_too_few_rows_in_a_domain_raises():
    rows = [r for r in _rows() if r["domain"] != "D03" or r["pair_id"] < "p03_03"]
    with pytest.raises(ValueError, match="D03"):
        build_splits(rows)


@pytest.fixture
def tiny_bench(tmp_path):
    rows = _rows()
    data = tmp_path / "bench.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data)
    write_splits(build_splits(rows), tmp_path)
    return data, tmp_path


def test_split_b_is_sealed(tiny_bench, monkeypatch):
    data, splits_dir = tiny_bench
    monkeypatch.delenv(bench.UNSEAL_ENV, raising=False)
    with pytest.raises(SealedSplitError):
        load_instances("B", path=data, directory=splits_dir)
    monkeypatch.setenv(bench.UNSEAL_ENV, "0")
    with pytest.raises(SealedSplitError):
        load_instances("B", path=data, directory=splits_dir)


def test_split_b_loads_only_when_unsealed(tiny_bench, monkeypatch):
    data, splits_dir = tiny_bench
    monkeypatch.setenv(bench.UNSEAL_ENV, "1")
    assert len(load_instances("B", path=data, directory=splits_dir)) == 50


def test_split_a_loads_with_gold(tiny_bench, monkeypatch):
    data, splits_dir = tiny_bench
    monkeypatch.delenv(bench.UNSEAL_ENV, raising=False)
    instances = load_instances("A", path=data, directory=splits_dir)
    assert len(instances) == 50
    assert all(x.gold.startswith("gold ") and x.gold == x.gold.strip() for x in instances)


def test_wrong_data_file_is_detected(tiny_bench, tmp_path):
    data, splits_dir = tiny_bench
    other = tmp_path / "other.parquet"
    rows = _rows()
    rows.reverse()
    pq.write_table(pa.Table.from_pylist(rows), other)
    with pytest.raises(ValueError, match="pair_id"):
        load_instances("A", path=other, directory=splits_dir)


def _instances(rows):
    return [
        Instance(
            id=instance_id(i),
            pair_id=r["pair_id"],
            arxiv_id=r["arxiv_id"],
            domain=r["domain"],
            paper1_title="t1",
            paper2_title="t2",
            abstract_1="a1",
            abstract_2="a2",
            gold="g",
        )
        for i, r in enumerate(rows)
    ]


def test_pick_baseline_covers_every_domain_and_is_reproducible():
    rows = _rows()
    a = [x for x in _instances(rows) if x.id in {i for i, _ in build_splits(rows)["A"]}]
    picked = pick_baseline(a)
    assert len(picked) == len(set(picked)) == 20
    assert len({x.domain for x in a if x.id in picked}) == 16
    assert picked == pick_baseline(list(reversed(a)))
    with pytest.raises(ValueError):
        pick_baseline(a, n=10)


def test_committed_split_files():
    ids = {s: split_ids(s) for s in "AB"}
    for s in "AB":
        assert len(ids[s]) == len({i for i, _ in ids[s]}) == 50
    assert not {i for i, _ in ids["A"]} & {i for i, _ in ids["B"]}
