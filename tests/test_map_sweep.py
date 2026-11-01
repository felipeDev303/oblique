"""The map sweep against a fake transport and embedder: nothing here makes a network call."""

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
import yaml
from tenacity import wait_none

from oblique import baseline, map_sweep
from oblique.bench import Instance
from oblique.deck import load_deck
from oblique.deck.controls import DECK_DIR, pick_controls
from oblique.generate.validate import InvalidReason
from oblique.map_sweep import MapSweep, classify_band, render_summary, select_ml
from oblique.smoke import BudgetExceeded

INSTANCES = [
    Instance(
        id=f"i-{1000 + j}",
        pair_id=f"p{j}",
        arxiv_id="2510.00001",
        domain=f"D{j:02d}",
        paper1_title="One",
        paper2_title="Two",
        abstract_1=f"a1 {j}",
        abstract_2="a2",
        gold="Gold.",
    )
    for j in range(6)
]
CARDS = {"seed": ["OB-001"], "paper": ["E6-1"], "control": ["CTRL-R-001"]}
FILTERED = "The response was filtered due to the prompt triggering our content management policy."


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def _error_400(message):
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=req), body=None)


class FakeAPI:
    """Ideas under OB-001 are long and beat the anchor; E6-1's are short and lose to it. Distance
    to S0 is high for OB-001, mid for E6-1, and low for everything else."""

    def __init__(self, refuse=None):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.requests = []
        self.refuse = refuse or (lambda system, user: False)

    async def _models(self):
        return SimpleNamespace(data=[SimpleNamespace(id="muse-spark-1.3-contributor")])

    async def create(self, **request):
        self.requests.append(request)
        system, user = (m["content"] for m in request["messages"])
        if self.refuse(system, user):
            raise _error_400(FILTERED)
        if "research scientist" in system:
            tag = re.search(r"«(.*?)»", user)
            card = tag.group(1) if tag else ""
            kind = (
                "long"
                if card.startswith("Remove")
                else "short"
                if card.startswith("Begin")
                else "mid"
            )
            words = {"long": 30, "mid": 20, "short": 10}[kind]
            content = json.dumps(
                {
                    "ideas": [
                        {
                            "problem_gap": f"[Evidence Gap] {kind} gap {len(self.requests)}",
                            "core_claim": " ".join(["word"] * words),
                            "method": "[Empirical Mapping] method",
                            "inspiration": f"i {len(self.requests)}",
                        }
                    ]
                }
            )
        elif "expert reviewer" in system:
            a, b = (len(s.split()) for s in re.findall(r"Claim: (.*)", user))
            content = json.dumps({"winner": "A" if a >= b else "B"})
        elif "field by field" in system:
            s = 1 if "long gap" in user else 3 if "short gap" in user else 5
            content = json.dumps({"problem_gap": s, "core_claim": s, "method": s})
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
        fields = ("problem_gap", "core_claim", "method")
        return [{f: np.array([1.0, len(i.core_claim) / 50]) for f in fields} for i in ideas]


@pytest.fixture
def cfg(tmp_path):
    c = yaml.safe_load(open("configs/R-20261031-01.yaml", encoding="utf-8"))
    c["cache"] = str(tmp_path / "cache.db")
    c["cards"] = CARDS
    c["n_boot"] = 50
    return c


def sweep(cfg, tmp_path, monkeypatch, api):
    monkeypatch.setattr(baseline, "load_instances", lambda split: INSTANCES)
    monkeypatch.setattr(map_sweep, "load_instances", lambda split: INSTANCES)
    s = MapSweep(
        cfg, out=tmp_path / "run", transport=api, wait=wait_none(), embedder=FakeEmbedder()
    )
    s.check_cards = lambda: None
    return s


def metrics(tmp_path):
    return json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))


def by_condition(result):
    return {r["condition"]: r for r in result["conditions"]}


def users(api, system_text):
    return [r for r in api.requests if system_text in r["messages"][0]["content"]]


def test_generation_covers_anchors_s0_and_every_condition(cfg, tmp_path, monkeypatch):
    api = FakeAPI()
    s = sweep(cfg, tmp_path, monkeypatch, api)
    asyncio.run(s.run())
    # 6 anchors + 6 x 3 S0 + 6 x 5 conditions
    gens = users(api, "research scientist")
    assert len(gens) == 6 + 18 + 30
    assert all("seed" not in r for r in gens)
    carded = [r for r in gens if "oblique constraint" in r["messages"][1]["content"]]
    assert len(carded) == 18
    deck = load_deck(DECK_DIR)
    assert sum(deck["OB-001"].text in r["messages"][1]["content"] for r in carded) == 6
    hot = [r for r in gens if r["temperature"] == 1.5]
    assert len(hot) == 6 and all(
        "oblique constraint" not in r["messages"][1]["content"] for r in hot
    )
    records = s.slots[(INSTANCES[0].id, "OB-001")]
    assert records[0].card == "OB-001" and "+" in records[0].prompt_hash
    assert s.slots[(INSTANCES[0].id, "C2")][0].card is None
    assert s.slots[(INSTANCES[0].id, "C1s")][2].slot == 2


def test_c2_never_shares_a_draw_with_the_anchor_or_s0(cfg, tmp_path, monkeypatch):
    s = sweep(cfg, tmp_path, monkeypatch, FakeAPI())
    asyncio.run(s.run())
    x = INSTANCES[0].id
    c2 = s.slots[(x, "C2")][0].idea
    assert c2 != s.anchors[x] and c2 != s.slots[(x, "C1s")][0].idea


def test_judges_compare_with_the_anchor_and_every_idea_of_s0(cfg, tmp_path, monkeypatch):
    api = FakeAPI()
    asyncio.run(sweep(cfg, tmp_path, monkeypatch, api).run())
    assert len(users(api, "expert reviewer")) == 2 * 30
    assert len(users(api, "You assess how much")) == 30
    assert len(users(api, "field by field")) == 3 * 30
    assert all(r["temperature"] == 0 for r in users(api, "field by field"))


def test_report_gives_every_metric_and_classifies_the_bands(cfg, tmp_path, monkeypatch):
    asyncio.run(sweep(cfg, tmp_path, monkeypatch, FakeAPI()).run())
    result = metrics(tmp_path)
    rows = by_condition(result)
    assert set(rows) == {"OB-001", "E6-1", "CTRL-R-001", "C2", "C2-T"}
    assert result["reference"]["d_c2"]["mean"] == 0.0
    long_card = rows["OB-001"]
    assert long_card["q_raw"]["mean"] == 1.0 and long_card["d"]["mean"] == 1.0
    assert long_card["m"]["mean"] == 1.0 and long_card["band"] == "liminal"
    assert long_card["confirmed"] is True
    assert long_card["grounding"]["mean"] == 0.75
    assert long_card["q_controlled"]["n"] == 6
    assert rows["E6-1"]["q_raw"]["mean"] == 0.0 and rows["E6-1"]["band"] == "destructive"
    assert rows["CTRL-R-001"]["band"] == "inert"
    assert rows["C2"]["band"] == "reference"
    assert rows["OB-001"]["invalid_rate"]["mean"] == 0.0
    assert rows["OB-001"]["filtered_rate"]["mean"] == 0.0
    assert result["bands"]["liminal"] == ["OB-001"]
    assert result["liminal_confirmed"] == ["OB-001"]
    # One liminal OB card is below the minimum of 3, so ML falls back to the non-destructive OB.
    assert result["ml"] == {"cards": ["OB-001"], "fallback": True, "liminal_ob": ["OB-001"]}
    assert rows["OB-001"]["q_controlled_minus_c2"] is not None
    assert rows["C2"]["q_raw_minus_c2"] is None
    assert result["spearman_d_llm_vs_embed"]["n"] == 90
    assert "OB-001" in (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert "OB-001" in render_summary(result)


def test_a_resume_pays_nothing_and_keeps_the_spend(cfg, tmp_path, monkeypatch):
    first = FakeAPI()
    run = sweep(cfg, tmp_path, monkeypatch, first)
    asyncio.run(run.run())
    spent = run.spent_usd
    assert spent > 0
    second = FakeAPI()
    again = sweep(cfg, tmp_path, monkeypatch, second)
    assert again.spent_usd == pytest.approx(spent)  # read back from the ledger
    asyncio.run(again.run())
    assert second.requests == []
    assert again.spent_usd == pytest.approx(spent)
    assert metrics(tmp_path)["spent_usd"] == pytest.approx(spent, abs=1e-4)


def test_a_stopped_run_resumes_where_it_was(cfg, tmp_path, monkeypatch):
    cfg["budget"]["stop_usd"] = 0.0002  # stops partway through
    first = FakeAPI()
    with pytest.raises(BudgetExceeded):
        asyncio.run(sweep(cfg, tmp_path, monkeypatch, first).run())
    paid = len(first.requests)
    assert 0 < paid < 54 + 180

    cfg["budget"]["stop_usd"] = 5
    second = FakeAPI()
    asyncio.run(sweep(cfg, tmp_path, monkeypatch, second).run())
    # What the first launch finished is not paid again (calls in flight when it stopped may be).
    assert len(second.requests) <= 54 + 180 - paid + 16
    assert metrics(tmp_path)["instances"] == 6


def test_the_ledger_counts_towards_the_cap_on_a_resume(cfg, tmp_path, monkeypatch):
    asyncio.run(sweep(cfg, tmp_path, monkeypatch, FakeAPI()).run())
    cfg["budget"]["stop_usd"] = 0.0
    cfg["run_id"] = cfg["run_id"]
    (tmp_path / "other").mkdir()
    cfg["cache"] = str(tmp_path / "other" / "cache.db")  # nothing cached: every call would pay
    api = FakeAPI()
    with pytest.raises(BudgetExceeded):
        asyncio.run(sweep(cfg, tmp_path, monkeypatch, api).run())
    assert users(api, "research scientist") == []


def test_a_filtered_generation_counts_as_q0_and_apart_from_invalid(cfg, tmp_path, monkeypatch):
    deck = load_deck(DECK_DIR)
    text = deck["OB-001"].text
    seen = set()

    def refuse(system, user):
        # One instance, the OB-001 idea only.
        if "research scientist" in system and text in user and "a1 0\n" in user:
            seen.add(user)
            return True
        return False

    s = sweep(cfg, tmp_path, monkeypatch, FakeAPI(refuse))
    asyncio.run(s.run())
    rows = by_condition(metrics(tmp_path))
    r = s.slots[(INSTANCES[0].id, "OB-001")][0]
    assert r.invalid_reason == InvalidReason.content_filtered
    assert rows["OB-001"]["filtered_rate"]["mean"] == pytest.approx(1 / 6)
    assert rows["OB-001"]["invalid_rate"]["mean"] == 0.0
    assert rows["OB-001"]["q_raw"]["n"] == 6  # q = 0 stays in the mean
    assert rows["OB-001"]["q_raw"]["mean"] == pytest.approx(5 / 6)
    assert rows["OB-001"]["d"]["n"] == 5
    assert rows["OB-001"]["m"]["n"] == 6
    assert metrics(tmp_path)["content_filtered"][0]["judge"] == "generation"


def test_a_filtered_judgment_is_missing_and_leaves_the_paired_comparison(
    cfg, tmp_path, monkeypatch
):
    def refuse(system, user):
        # Distance between one instance's C2-T idea and one of its S0 ideas is never available.
        return False

    s = sweep(cfg, tmp_path, monkeypatch, FakeAPI(refuse))
    original = s.distance_pair

    async def distance_pair(client, x, cond, i, j, a, b):
        if x.id == INSTANCES[0].id and cond == "OB-001" and j == 1:
            s.d_llm[(x.id, cond, i, j)] = None
            s.content_filtered.append({"judge": "distance-spark", "instance": x.id})
            return
        await original(client, x, cond, i, j, a, b)

    s.distance_pair = distance_pair
    asyncio.run(s.run())
    result = metrics(tmp_path)
    rows = by_condition(result)
    assert rows["OB-001"]["d"]["n"] == 5  # min over S0 needs all three distances
    assert rows["OB-001"]["d_minus_c2"]["n"] == 5
    assert rows["OB-001"]["q_raw"]["n"] == 6
    assert rows["OB-001"]["m"]["n"] == 5
    assert result["missing_values"]["d"] == 1


def test_a_filtered_anchor_leaves_its_instance_out_of_q(cfg, tmp_path, monkeypatch):
    s = sweep(cfg, tmp_path, monkeypatch, FakeAPI())
    x = INSTANCES[0].id
    original = s.generate_anchors

    async def generate_anchors(client, instances):
        await original(client, instances)
        s.anchors[x] = None

    s.generate_anchors = generate_anchors
    asyncio.run(s.run())
    result = metrics(tmp_path)
    rows = by_condition(result)
    assert result["anchors_valid"] == 5
    assert rows["OB-001"]["q_raw"]["n"] == 5 and rows["OB-001"]["d"]["n"] == 6
    assert rows["OB-001"]["q_controlled"]["n"] == 5


@pytest.mark.parametrize(
    "dq_lo, dq_hi, dd_lo, dd_hi, band, confirmed",
    [
        (-0.02, 0.10, 0.05, 0.30, "liminal", True),  # far, and q is clearly not lower
        (-0.20, 0.10, 0.05, 0.30, "liminal", False),  # far, but the CI of q reaches below -delta
        (-0.20, -0.10, 0.05, 0.30, "destructive", False),  # far, q falls
        (-0.20, -0.10, -0.10, 0.10, "destructive", False),  # priority over inert
        (-0.20, -0.10, -0.30, -0.10, "destructive", False),  # priority over convergent
        (-0.02, 0.10, -0.10, 0.10, "inert", False),  # the CI of d contains 0
        (-0.02, 0.10, 0.00, 0.10, "inert", False),  # a bound exactly at 0 contains it
        (-0.02, 0.10, -0.30, -0.10, "convergent", False),  # closer than a re-sample
        (-0.20, -0.05, 0.05, 0.30, "liminal", False),  # hi exactly at -delta is not below it
        (-0.05, 0.10, 0.05, 0.30, "liminal", False),  # lo exactly at -delta is not above it
    ],
)
def test_classify_band(dq_lo, dq_hi, dd_lo, dd_hi, band, confirmed):
    out = classify_band(dq_lo, dq_hi, dd_lo, dd_hi, 0.05)
    assert out["band"] == band and out["confirmed"] is confirmed
    assert out["destructive"] is (band == "destructive")


def _card(condition, band, m, origin="seed"):
    return {"condition": condition, "band": band, "origin": origin, "m": {"mean": m}}


def test_ml_is_every_liminal_ob_card():
    rows = [
        _card("OB-001", "liminal", 0.1),
        _card("OB-002", "liminal", 0.2),
        _card("OB-003", "liminal", 0.3),
        _card("OB-004", "inert", 0.9),
        _card("E6-1", "liminal", 0.9, "paper"),
        _card("CTRL-R-001", "liminal", 0.9, "control"),
        _card("C2-T", "liminal", 0.9, "no_card"),
    ]
    assert select_ml(rows) == {
        "cards": ["OB-001", "OB-002", "OB-003"],
        "fallback": False,
        "liminal_ob": ["OB-001", "OB-002", "OB-003"],
    }


def test_ml_falls_back_to_the_six_non_destructive_ob_cards_with_highest_m():
    rows = [_card(f"OB-{i:03d}", "inert", i / 100) for i in range(1, 10)]
    rows += [
        _card("OB-050", "destructive", 0.99),
        _card("OB-051", "liminal", 0.001),
        _card("OB-052", "liminal", 0.002),
        _card("E6-1", "inert", 0.99, "paper"),
        _card("CTRL-R-001", "inert", 0.99, "control"),
    ]
    out = select_ml(rows)
    assert out["fallback"] is True
    assert out["liminal_ob"] == ["OB-051", "OB-052"]
    assert out["cards"] == ["OB-009", "OB-008", "OB-007", "OB-006", "OB-005", "OB-004"]


def test_committed_config_lists_the_deck_and_the_seeded_controls(tmp_path):
    cfg = yaml.safe_load(open("configs/R-20261031-01.yaml", encoding="utf-8"))
    deck = load_deck(DECK_DIR)
    by_origin = {
        o: sorted(c.id for c in deck.values() if c.origin == o)
        for o in ("seed", "paper", "control")
    }
    assert cfg["cards"]["seed"] == by_origin["seed"] and len(cfg["cards"]["seed"]) == 30
    assert cfg["cards"]["paper"] == by_origin["paper"] and len(cfg["cards"]["paper"]) == 6
    assert cfg["cards"]["control"] == pick_controls(by_origin["control"], 10)
    assert cfg["no_card"] == {"C2": {}, "C2-T": {"temperature": 1.5}}
    assert cfg["judges"]["spark"]["concurrency"] == 8
    assert cfg["judges"]["spark"]["max_concurrency"] == 12
    cfg["cache"] = str(tmp_path / "cache.db")
    s = MapSweep(cfg, out=tmp_path / "run")
    s.check_cards()
    assert len(s.conditions) == 48


def test_check_cards_refuses_a_hand_picked_control(tmp_path):
    cfg = yaml.safe_load(open("configs/R-20261031-01.yaml", encoding="utf-8"))
    cfg["cache"] = str(tmp_path / "cache.db")
    other = "CTRL-R-002" if "CTRL-R-002" not in cfg["cards"]["control"] else "CTRL-R-003"
    cfg["cards"]["control"] = [*cfg["cards"]["control"][:-1], other]
    with pytest.raises(SystemExit, match="seeded draw"):
        MapSweep(cfg, out=tmp_path / "run").check_cards()


def test_plan_counts_calls_and_sees_what_is_cached(cfg, tmp_path, monkeypatch):
    s = sweep(cfg, tmp_path, monkeypatch, FakeAPI())
    plan = s.plan(INSTANCES)
    assert plan["ideas"] == 30
    assert plan["blocks"]["condition"] == {
        "calls": 30,
        "cached": 0,
        "to_pay": 30,
        "usd": pytest.approx(30 * cfg["estimate"]["condition"]["usd"]),
    }
    assert plan["blocks"]["quality"]["calls"] == 60
    assert plan["blocks"]["distance"]["calls"] == 90
    assert plan["calls_total"] == 6 + 18 + 30 + 60 + 30 + 90
    assert plan["judge_minutes_at_ceiling"] < plan["judge_minutes_at_start_limit"]

    asyncio.run(s.run())
    after = sweep(cfg, tmp_path, monkeypatch, FakeAPI()).plan(INSTANCES)
    assert after["blocks"]["anchor"]["cached"] == 6
    assert after["blocks"]["s0"]["cached"] == 18
    assert after["blocks"]["condition"]["cached"] == 30
