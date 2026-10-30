"""The smoke run end to end against a fake transport: nothing here makes a network call."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml

from oblique import smoke
from oblique.bench import Instance

INSTANCE = Instance(
    id="i-0032",
    pair_id="p",
    arxiv_id="2601.15561",
    domain="CS-Other",
    paper1_title="One",
    paper2_title="Two",
    abstract_1="a1",
    abstract_2="a2",
    gold="Gold insight.",
)
IDEA = {
    "problem_gap": "[Evidence Gap] g",
    "core_claim": "c",
    "method": "[Empirical Mapping] m",
    "inspiration": "i",
}


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeAPI:
    """Rejects reasoning_effort="none"; answers each prompt with its output contract."""

    MODELS = ("muse-spark-1.3", "muse-spark-1.3-contributor", "muse-wonder-1.0")

    def __init__(self, models=MODELS, completion_tokens=10):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.available = models
        self.completion_tokens = completion_tokens
        self.requests = []

    async def _models(self):
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in self.available])

    async def create(self, **request):
        self.requests.append(request)
        if request.get("reasoning_effort") == "none":
            req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
            raise openai.BadRequestError(
                "bad effort", response=httpx.Response(400, request=req), body=None
            )
        system = request["messages"][0]["content"]
        if "research scientist" in system:
            content = json.dumps({"ideas": [IDEA] * 3})
        elif "expert reviewer" in system:
            content = '{"winner": "A"}'
        elif "field by field" in system:
            content = '{"problem_gap": 4, "core_claim": 3, "method": 2}'
        else:
            content = '{"score": 4}'
        return _Raw(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": self.completion_tokens},
            }
        )


def _cfg(tmp_path):
    cfg = yaml.safe_load(open("configs/R-20261030-01.yaml", encoding="utf-8"))
    cfg["cache"] = str(tmp_path / "cache.db")
    return cfg


def test_identical_modes_abort_before_any_call(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["generator"]["standard"] = cfg["generator"]["contributor"]
    api = FakeAPI()
    with pytest.raises(SystemExit, match="identical"):
        asyncio.run(smoke.Smoke(cfg, out=tmp_path / "run", transport=api).run())
    assert api.requests == []


def test_missing_model_stops_before_any_paid_call(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "load_instances", lambda split: [INSTANCE])
    api = FakeAPI(models=("muse-spark-1.3", "muse-wonder-1.0"))
    run = smoke.Smoke(_cfg(tmp_path), out=tmp_path / "run", transport=api)
    with pytest.raises(smoke.MissingModels, match="muse-spark-1.3-contributor"):
        asyncio.run(run.run())
    assert api.requests == []
    assert run.notes["models_missing"] == ["muse-spark-1.3-contributor"]


def test_budget_stops_new_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "load_instances", lambda split: [INSTANCE])
    # 40k output tokens at 4.25/M = US$0.17 per call: the 4th call is refused.
    api = FakeAPI(completion_tokens=40_000)
    run = smoke.Smoke(_cfg(tmp_path), out=tmp_path / "run", transport=api)
    with pytest.raises(smoke.BudgetExceeded):
        asyncio.run(run.run())
    assert len(run.calls) == 3 and run.spent_usd > 0.5


def test_smoke_runs_end_to_end(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(smoke, "load_instances", lambda split: [INSTANCE])
    api = FakeAPI()
    run = smoke.Smoke(cfg, out=tmp_path / "run", transport=api)
    asyncio.run(run.run())

    notes = run.notes
    assert notes["json_mode_accepted"] is True
    assert notes["parse g1"]["valid"] == 3 and notes["parse g1"]["shape"] == "ideas_object"
    assert notes["seed"]["same_seed_same_text"] is True
    assert notes["quality spark"] == {"idea_first": "A", "gold_first": "A", "q": 0.5}
    assert notes["quality wonder"]["q"] == 0.5
    assert notes["quality spark idea_first: reasoning_effort"]["accepted"] == "minimal"
    assert notes["distance spark"]["d"] == (1 / 4 + 2 / 4 + 3 / 4) / 3
    assert notes["grounding spark"]["grounding"] == 0.75

    # 5 generations + 4 Spark judgments + 2 Wonder judgments; rejected requests are not logged.
    assert len(run.calls) == 11
    assert sum(r.get("reasoning_effort") == "none" for r in api.requests) == 2
    gen = [r for r in api.requests if "research scientist" in r["messages"][0]["content"]]
    assert [r["seed"] for r in gen] == [0, 0, 1, 0, 0]
    assert [r["model"] for r in gen] == ["muse-spark-1.3-contributor"] * 3 + ["muse-spark-1.3"] * 2
    assert "reasoning_effort" not in gen[3] and gen[4]["reasoning_effort"] == "low"
    assert set(notes["generation reasoning tokens"]) == {"g1", "g2", "g3", "g4", "g5"}
    for name in ("generations", "judgments", "calls"):
        assert (tmp_path / "run" / f"{name}.jsonl").stat().st_size > 0

    # A rerun is served from the cache; only the (unpaid, uncached) rejected probes repeat.
    api.requests.clear()
    asyncio.run(smoke.Smoke(cfg, out=tmp_path / "rerun", transport=api).run())
    assert all(r.get("reasoning_effort") == "none" for r in api.requests)
