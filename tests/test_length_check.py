"""The ADR-009 validation run against a fake transport and embedder: no network."""

import asyncio
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from tenacity import wait_none

from oblique import baseline, length_check
from oblique.bench import Instance
from oblique.length_check import distribution
from oblique.prompts import IDEATION_V3, PROMPTS

INSTANCES = [
    Instance(
        id=f"i-{1000 + j}",
        pair_id=f"p{j}",
        arxiv_id="2510.00001",
        domain=f"D{j % 16:02d}",
        paper1_title="One",
        paper2_title="Two",
        abstract_1=f"a1 {j}",
        abstract_2="a2",
        gold="Gold insight.",
    )
    for j in range(50)
]


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeAPI:
    """K = 1 ideas have long claims, K = 3 ideas short ones, so C1 is shorter than the anchor.
    The judge prefers the longer idea; on a tie, position A."""

    def __init__(self):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.requests = []

    async def _models(self):
        return SimpleNamespace(data=[SimpleNamespace(id="muse-spark-1.3-contributor")])

    def idea(self, words: int, slot: int) -> dict:
        n = f"{len(self.requests)}-{slot}"
        return {
            "problem_gap": f"[Evidence Gap] gap {n}",
            "core_claim": " ".join(["word"] * words) + f" {n}",
            "method": "[Empirical Mapping] method",
            "inspiration": "i",
        }

    async def create(self, **request):
        self.requests.append(request)
        system, user = (m["content"] for m in request["messages"])
        if "research scientist" in system:
            k = 3 if "exactly 3 objects" in user else 1
            content = json.dumps({"ideas": [self.idea(5 if k == 3 else 20, s) for s in range(k)]})
        elif "expert reviewer" in system:
            a, b = (len(s.split()) for s in re.findall(r"Claim: (.*)", user))
            content = json.dumps({"winner": "A" if a >= b else "B"})
        else:
            content = '{"score": 4}'
        return _Raw(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )


class FakeEmbedder:
    def embed(self, ideas):
        return [
            {f: np.array([1.0, float(i % 5)]) for f in ("problem_gap", "core_claim", "method")}
            for i in range(len(ideas))
        ]


@pytest.fixture
def run(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/R-20261030-05.yaml", encoding="utf-8"))
    cfg["cache"] = str(tmp_path / "cache.db")
    cfg["instances"] = baseline.pick_baseline(INSTANCES)
    cfg["n_boot"] = 50
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    api = FakeAPI()
    check = length_check.LengthCheck(
        cfg, out=tmp_path / "run", transport=api, wait=wait_none(), embedder=FakeEmbedder()
    )
    asyncio.run(check.run())
    return api, check, json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))


def generations(api):
    return [r for r in api.requests if "research scientist" in r["messages"][0]["content"]]


def test_run_generates_every_condition_with_v3(run):
    api, check, result = run
    gens = generations(api)
    # 20 anchors + 60 C0 + 20 C1 + 60 C1s
    assert len(gens) == 160
    assert all(r["messages"][0]["content"] == IDEATION_V3.system for r in gens)
    assert all("seed" not in r for r in gens)
    records = [r for recs in check.slots.values() for r in recs]
    assert len(records) == 180 and {r.condition for r in records} == {"C0", "C1", "C1s"}
    assert {r.prompt_hash for r in records} == {IDEATION_V3.hash}
    assert result["anchors_valid"] == 20
    assert result["invalid_ideas"] == {"C0": 0, "C1": 0, "C1s": 0}


def test_sequential_ideas_are_conditioned_on_the_earlier_ones(run):
    api, check, _ = run
    users = [r["messages"][1]["content"] for r in generations(api)]
    assert sum("Idea 1:" in u and "Idea 2:" not in u for u in users) == 20  # second idea of a set
    assert sum("Idea 2:" in u for u in users) == 20  # third idea of a set
    for x in check.cfg["instances"]:
        records = check.slots[(x, "C1s")]
        third = next(u for u in users if "Idea 2:" in u and records[0].idea.core_claim in u)
        assert records[1].idea.core_claim in third
        assert [r.slot for r in records] == [0, 1, 2]


def test_a_sequence_never_shares_a_draw_with_c0_or_the_anchor(run):
    _, check, _ = run
    # The first idea of a sequence reads like a C0 call, so only sample_idx keeps them apart.
    claims = {
        c: {r.idea.core_claim for (_, cc), rs in check.slots.items() if cc == c for r in rs}
        for c in check.conditions
    }
    assert not claims["C0"] & claims["C1s"]
    assert not {a.core_claim for a in check.anchors.values()} & (claims["C0"] | claims["C1s"])


def test_judges_use_quality_v2_and_grounding(run):
    api, _, result = run
    quality = [r for r in api.requests if "expert reviewer" in r["messages"][0]["content"]]
    assert len(quality) == 360
    assert all(r["messages"][0]["content"] == PROMPTS["quality-v2"].system for r in quality)
    assert all(r["reasoning_effort"] == "minimal" and r["temperature"] == 0 for r in quality)
    assert result["grounding"]["C1s"]["mean"] == 0.75
    assert "C1s-C0" in result["grounding"] and "C1s-C1" in result["grounding"]


def test_report_has_length_q_and_controlled_rates(run):
    _, _, result = run
    words = result["length"]
    assert words["C1"]["words"]["mean"] < words["C0"]["words"]["mean"]
    assert words["ANCHOR"]["words"]["n"] == 20
    assert words["C0"]["within_caps"]["core_claim"] == 1.0

    q = result["q_anchor"]
    assert set(q) >= {"C0", "C1", "C1s", "q C1-C0", "q C1s-C0", "q C1s-C1", "position_all"}
    assert q["C1"]["q"]["mean"] == 0.0  # always shorter than the anchor
    assert result["length_vs_q"]["fallback_threshold"] == 0.3
    assert result["length_vs_q"]["all"]["n"] == 180
    assert isinstance(result["length_vs_q"]["fallback_triggered"], bool)

    controlled = result["controlled_for_length"]
    assert controlled["judgments"] == 360
    assert set(controlled["rates"]) == {"C0", "C1", "C1s"}
    assert set(controlled["differences"]) == {"C1-C0", "C1s-C0", "C1s-C1"}
    assert result["embedding_distance"]["C1s"]["n"] == 20


def test_distribution():
    d = distribution([1, 2, 3, 4, 5])
    assert d == {
        "n": 5,
        "mean": 3.0,
        "sd": d["sd"],
        "min": 1,
        "p25": 2.0,
        "median": 3.0,
        "p75": 4.0,
        "max": 5,
    }
    assert d["sd"] == pytest.approx(1.5811, abs=1e-3)
