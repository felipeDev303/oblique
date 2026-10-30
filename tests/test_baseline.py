"""The baseline run end to end against a fake transport and a fake embedder: no network."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from tenacity import wait_none

from oblique import baseline
from oblique.bench import Instance

DOMAINS = [f"D{i:02d}" for i in range(16)]
INSTANCES = [
    Instance(
        id=f"i-{1000 + j}",
        pair_id=f"p{j}",
        arxiv_id="2510.00001",
        domain=DOMAINS[j % 16],
        paper1_title="One",
        paper2_title="Two",
        abstract_1=f"a1 {j}",
        abstract_2="a2",
        gold="Gold insight paragraph.",
    )
    for j in range(50)
]


def idea(n: int) -> dict:
    return {
        "problem_gap": f"[Evidence Gap] gap {n}",
        "core_claim": f"claim {n}",
        "method": f"[Empirical Mapping] method {n}",
        "inspiration": "i",
    }


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeAPI:
    """Every set is valid except the first C0 call, which is not JSON (and is retried).
    Wonder always prefers position A; Spark always prefers the generated idea."""

    def __init__(self):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.requests = []
        self.broke_once = False

    async def _models(self):
        ids = ["muse-spark-1.3", "muse-spark-1.3-contributor", "muse-wonder-1.0"]
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in ids])

    async def create(self, **request):
        self.requests.append(request)
        system, user = (m["content"] for m in request["messages"])
        if "research scientist" in system:
            k = 1 if "exactly 1 object" in user else 3
            if k == 1 and not self.broke_once:
                self.broke_once = True
                content = "not json"
            else:
                content = json.dumps(
                    {"ideas": [idea(len(self.requests) * 10 + i) for i in range(k)]}
                )
        elif "expert reviewer" in system:
            # Wonder and the anchor comparison always prefer position A, so q = 0.5.
            if request["model"] == "muse-wonder-1.0" or "Gold" not in user:
                content = '{"winner": "A"}'
            else:
                content = (
                    '{"winner": "A"}'
                    if user.index("Problem:") < user.index("Gold")
                    else '{"winner": "B"}'
                )
        elif "field by field" in system:
            content = '{"problem_gap": 2, "core_claim": 3, "method": 4}'
        else:
            content = '{"score": 3}'
        return _Raw(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )


class FakeEmbedder:
    def embed(self, ideas):
        out = []
        for idea in ideas:
            n = int(idea.core_claim.split()[-1])
            out.append(
                {f: np.array([1.0, n % 7 + 1.0]) for f in ("problem_gap", "core_claim", "method")}
            )
        return out


@pytest.fixture
def cfg(tmp_path):
    import yaml

    c = yaml.safe_load(open("configs/R-20261030-03.yaml", encoding="utf-8"))
    c["cache"] = str(tmp_path / "cache.db")
    c["instances"] = baseline.pick_baseline(INSTANCES)
    return c


def test_baseline_end_to_end(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    api = FakeAPI()
    run = baseline.Baseline(
        cfg, out=tmp_path / "run", transport=api, wait=wait_none(), embedder=FakeEmbedder()
    )
    asyncio.run(run.run())
    result = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))
    m = result["metrics"]

    # 20 x (3 C0 + 1 C1) generations + 1 retry after the broken C0 call.
    gens = [r for r in api.requests if "research scientist" in r["messages"][0]["content"]]
    assert len(gens) == 81
    assert all("seed" not in r for r in api.requests)
    assert m["paired_instances"] == 20 and m["incomplete_sets"] == 0
    for cond in ("C0", "C1"):
        cell = m["table"][cond]
        assert cell["n_sets"] == 20 and cell["invalid_ideas"] == 0
        assert cell["q"]["mean"] == 1.0  # Spark always prefers the idea
        assert cell["d"]["mean"] == pytest.approx(((5 - 2) + (5 - 3) + (5 - 4)) / 12)
        assert cell["grounding"]["mean"] == 0.5

    cal = m["calibration"]
    assert cal["position_spark"]["consistent"] == 1.0
    assert cal["position_wonder"]["a_both"] == 1.0
    assert cal["spark_vs_wonder_q"]["n"] == 120
    assert cal["spark_vs_wonder_q"]["exact"] == 0.0  # Spark q = 1, Wonder q = 0.5
    assert cal["spearman_d_llm_vs_embed"]["n"] == 120
    assert set(result["examples"]) == {"C0", "C1"} and len(result["examples"]["C1"]) == 2

    lines = (tmp_path / "run" / "generations.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 120
    c0 = [json.loads(x) for x in lines if '"C0"' in x]
    assert sorted({r["slot"] for r in c0}) == [0, 1, 2]
    assert sum(r["attempt"] == 1 for r in c0) == 1
    assert all(r["seed"] is None for r in c0)


def test_anchor_is_the_primary_q_and_the_gold_is_secondary(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    cfg["q_reference"] = "anchor"
    api = FakeAPI()
    run = baseline.Baseline(
        cfg, out=tmp_path / "run", transport=api, wait=wait_none(), embedder=FakeEmbedder()
    )
    asyncio.run(run.run())
    m = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))["metrics"]

    gens = [r for r in api.requests if "research scientist" in r["messages"][0]["content"]]
    assert len(gens) == 81 + 20  # one anchor per instance
    assert len((tmp_path / "run" / "anchors.jsonl").read_text("utf-8").splitlines()) == 20
    for cond in ("C0", "C1"):
        assert m["table"][cond]["q"]["mean"] == 0.5
        assert m["table"][cond]["q_gold"]["mean"] == 1.0
        assert m["q_report"]["anchor"][cond]["tie_share"] == 1.0
        # The anchor gives CU its q: with q = 0.5 throughout, CU sits below the gold's.
        assert m["table"][cond]["cu"]["mean"] < 0.5
    assert m["calibration"]["position_anchor"]["a_both"] == 1.0


def test_config_instances_must_match_the_seeded_pick(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    cfg["instances"] = cfg["instances"][:-1] + ["i-9999"]
    run = baseline.Baseline(cfg, out=tmp_path / "run", transport=FakeAPI())
    with pytest.raises(SystemExit, match="pick_baseline"):
        asyncio.run(run.run())
