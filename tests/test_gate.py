"""The baseline cost gate against a fake transport: nothing here makes a network call."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml
from tenacity import wait_none

from oblique import gate
from oblique.bench import Instance
from oblique.gate import latency_summary

IDEA = {
    "problem_gap": "[Evidence Gap] g",
    "core_claim": "c",
    "method": "[Empirical Mapping] m",
    "inspiration": "i",
}
IDS = ["i-0246", "i-0258"] + [f"i-{9000 + j}" for j in range(48)]
INSTANCES = [
    Instance(
        id=i,
        pair_id=i,
        arxiv_id="2510.00001",
        domain="ML/AI",
        paper1_title="One",
        paper2_title="Two",
        abstract_1=f"a1 {i}",
        abstract_2="a2",
        gold="g",
    )
    for i in IDS
]


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def _error(cls, status):
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return cls("x", response=httpx.Response(status, request=req), body=None)


class FakeAPI:
    """Accepts temperature <= 1.3; the first request of each burst of 16 gets a 429."""

    def __init__(self):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.requests = []
        self.active = self.max_active = 0
        self.limited = set()

    async def _models(self):
        return SimpleNamespace(data=[SimpleNamespace(id="muse-spark-1.3-contributor")])

    async def create(self, **request):
        self.requests.append(request)
        if request["temperature"] > 1.3:
            raise _error(openai.BadRequestError, 400)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            burst = (self.active, len(self.requests) // 16)
            if self.max_active >= 16 and self.active == 16 and burst not in self.limited:
                self.limited.add(burst)
                raise _error(openai.RateLimitError, 429)
        finally:
            self.active -= 1
        user = request["messages"][1]["content"]
        k = 1 if "exactly 1 object" in user else 3
        cached = (
            900
            if any("exactly 3" in r["messages"][1]["content"] for r in self.requests[:-1])
            else 0
        )
        return _Raw(
            {
                "choices": [{"message": {"content": json.dumps({"ideas": [IDEA] * k})}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        )


@pytest.fixture
def run(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/R-20261030-02.yaml", encoding="utf-8"))
    cfg["cache"] = str(tmp_path / "cache.db")
    monkeypatch.setattr(gate, "load_instances", lambda split: INSTANCES)
    api = FakeAPI()
    return gate.Gate(cfg, out=tmp_path / "run", transport=api, wait=wait_none()), api


def test_gate_end_to_end(run):
    g, api = run
    asyncio.run(g.run())
    notes = g.notes

    assert notes["cache"]["instance"] == "i-0246"
    assert notes["cache"]["second"]["cached"] == 900

    assert notes["temperature"]["max_accepted"] == 1.3
    assert [p["valid"] for p in notes["temperature"]["parses"]] == [3, 3]
    assert notes["temperature probe: temperature"]["accepted"] == 1.3

    rows = notes["concurrency"]
    assert [(r["level"], r["wave"]) for r in rows] == [
        (n, w) for n in (4, 8, 12, 16) for w in (0, 1)
    ]
    assert all(r["ok"] == r["requests"] and r["failed"] == 0 for r in rows)
    assert all(r["valid_sets"] == r["requests"] and r["sets_per_min"] > 0 for r in rows)
    assert notes["top_level_sets_per_min"] > 0
    assert api.max_active <= 16
    assert sum(r["rate_limited_429"] for r in rows if r["level"] == 16) >= 1
    assert all("seed" not in r for r in api.requests)
    # 2 cache + 2 temperature + 80 concurrency calls are logged; rejected probes are not.
    assert len(g.calls) == 84


def test_sustained_429s_at_the_top_level_mean_contributor_does_not_hold(run):
    g, api = run
    asyncio.run(g.run())
    top = [r for r in g.notes["concurrency"] if r["level"] == 16]
    expected = not all(r["rate_limited_429"] > 0 for r in top)
    assert g.notes["contributor_holds_top_level"] is expected


def test_latency_summary():
    assert latency_summary([]) == {"p50": None, "p90": None, "max": None}
    s = latency_summary([float(x) for x in range(1, 11)])
    assert s == {"p50": 5.5, "p90": 9.0, "max": 10.0}
