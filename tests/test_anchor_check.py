"""The anchor validation run against a fake transport: nothing here makes a network call."""

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml
from tenacity import wait_none

from oblique import anchor_check, baseline
from oblique.bench import Instance
from oblique.generate.records import records_from_set, write_jsonl
from oblique.generate.validate import parse_set
from oblique.llm import Usage
from oblique.prompts import IDEATION_V2

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
ANCHOR_SCORE = 50
SCORES = {"C0": [10, 50, 60], "C1": [60, 70, 80]}  # 50 ties with the anchor


# The quality prompt carries no paper text, so ideas must differ or requests hit the cache.
def idea(score: int, tag: str) -> dict:
    return {
        "problem_gap": f"[Evidence Gap] gap {tag}",
        "core_claim": f"score {score}",
        "method": "[Empirical Mapping] m",
        "inspiration": "i",
    }


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class FakeAPI:
    """Anchors score 50. The judge prefers the higher score; on a tie, position A.
    The first judge request gets one 429."""

    def __init__(self):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.requests = []
        self.limited = False

    async def _models(self):
        return SimpleNamespace(data=[SimpleNamespace(id="muse-spark-1.3-contributor")])

    async def create(self, **request):
        self.requests.append(request)
        system, user = (m["content"] for m in request["messages"])
        if "research scientist" in system:
            tag = re.search(r"a1 \d+", user).group()
            content = json.dumps({"ideas": [idea(ANCHOR_SCORE, f"anchor {tag}")]})
        else:
            if not self.limited:
                self.limited = True
                req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
                raise openai.RateLimitError(
                    "x", response=httpx.Response(429, request=req), body=None
                )
            a, b = (int(s) for s in re.findall(r"Claim: score (\d+)", user))
            content = json.dumps({"winner": "A" if a >= b else "B"})
        return _Raw(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )


@pytest.fixture
def cfg(tmp_path):
    c = yaml.safe_load(open("configs/R-20261030-04.yaml", encoding="utf-8"))
    c["cache"] = str(tmp_path / "cache.db")
    c["instances"] = baseline.pick_baseline(INSTANCES)
    records = []
    for iid in c["instances"]:
        for cond, scores in SCORES.items():
            parsed = parse_set(
                json.dumps({"ideas": [idea(s, f"{iid} {cond} {s}") for s in scores]}), 3
            )
            records += records_from_set(
                parsed,
                cards=[None] * 3,
                run_id="R-20261030-03",
                instance=iid,
                condition=cond,
                model="muse-spark-1.3-contributor",
                prompt_hash=IDEATION_V2.hash,
                seed=None,
                call_key="k",
                usage=Usage(),
            )
    source = tmp_path / "source.jsonl"
    write_jsonl(source, records)
    c["source_generations"] = str(source)
    return c


def run_check(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    api = FakeAPI()
    run = anchor_check.AnchorCheck(cfg, out=tmp_path / "run", transport=api, wait=wait_none())
    asyncio.run(run.run())
    return api, json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))


def test_anchor_check_end_to_end(cfg, tmp_path, monkeypatch):
    api, result = run_check(cfg, tmp_path, monkeypatch)
    gens = [r for r in api.requests if "research scientist" in r["messages"][0]["content"]]
    judged = [r for r in api.requests if "expert reviewer" in r["messages"][0]["content"]]
    assert len(gens) == 20  # anchors only: no new C0 / C1 ideas
    assert all("seed" not in r for r in api.requests)
    assert len(judged) == 240 + 1  # 120 ideas x 2 orders + the request that got the 429
    assert result["anchors_valid"] == 20

    q = result["q_anchor"]
    assert q["C0"]["q"]["mean"] == pytest.approx((0 + 0.5 + 1) / 3)
    assert q["C1"]["q"]["mean"] == 1.0
    assert q["q C1-C0"]["mean"] == pytest.approx(0.5)
    assert q["C0"]["tie_share"] == pytest.approx(1 / 3)
    assert q["C1"]["tie_share"] == 0.0
    # A tie is the same position winning both orders.
    assert q["C0"]["position"]["a_both"] == pytest.approx(1 / 3)
    assert q["C0"]["tie_share"] == pytest.approx(1 - q["C0"]["position"]["consistent"])
    assert q["position_all"]["n"] == 120

    judge = result["clients"]["judge"]
    assert judge["limit_reductions"] == 1 and judge["final_concurrency_limit"] == 4
    assert result["clients"]["generator"]["final_concurrency_limit"] == 16


def test_generators_and_judges_have_separate_limits(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    run = anchor_check.AnchorCheck(cfg, out=tmp_path / "run", transport=FakeAPI())
    gen = cfg["generator"]
    generator, judge = (
        run.new_client(gen["provider"], gen["concurrency"]),
        run.judge_client("spark"),
    )
    assert generator.limiter is not judge.limiter
    assert judge.limiter.limit == 8 and judge.limiter.adaptive is True
    assert generator.limiter.limit == 16 and generator.limiter.adaptive is False


def test_an_invalid_anchor_leaves_its_instance_without_q(cfg, tmp_path, monkeypatch):
    class NoAnchor(FakeAPI):
        broke = False

        async def create(self, **request):
            if "research scientist" in request["messages"][0]["content"] and not self.broke:
                self.broke = True
                self.requests.append(request)
                return _Raw(
                    {
                        "choices": [{"message": {"content": '{"ideas": [{}]}'}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                )
            return await super().create(**request)

    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    run = anchor_check.AnchorCheck(
        cfg, out=tmp_path / "run", transport=NoAnchor(), wait=wait_none()
    )
    asyncio.run(run.run())
    result = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))
    assert result["anchors_valid"] == 19
    assert result["q_anchor"]["C1"]["q"]["n"] == 19


def test_source_from_another_prompt_is_refused(cfg, tmp_path, monkeypatch):
    path = cfg["source_generations"]
    lines = open(path, encoding="utf-8").read().replace(IDEATION_V2.hash, "other")
    open(path, "w", encoding="utf-8").write(lines)
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    run = anchor_check.AnchorCheck(cfg, out=tmp_path / "run", transport=FakeAPI())
    with pytest.raises(SystemExit, match="another prompt"):
        asyncio.run(run.run())
