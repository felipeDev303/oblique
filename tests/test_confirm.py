"""The confirmation runner against a fake API and fake instances. Nothing here opens split B."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
import yaml
from tenacity import wait_none

from oblique import bench, confirm
from oblique.bench import Instance, SealedSplitError, split_ids
from oblique.cache import Cache
from oblique.confirm import ConfirmB, draw_cards, draw_wonder_positions, h1_verdict, main
from oblique.deck import load_deck
from oblique.deck.controls import DECK_DIR
from oblique.llm import ContentFilteredError, LLMClient
from oblique.metrics.bootstrap import bca_interval, bca_mean_ci, bootstrap_mean_ci
from oblique.metrics.calibration import controlled_win_rates

REAL_CONFIG = "configs/R-20261103-01.yaml"
IDS = [f"i-{1000 + j}" for j in range(6)]
INSTANCES = [
    Instance(
        id=iid,
        pair_id=f"p{j}",
        arxiv_id="2510.00001",
        domain=f"D{j:02d}",
        paper1_title="One",
        paper2_title="Two",
        abstract_1=f"a1 {j}",
        abstract_2="a2",
        gold="Gold.",
    )
    for j, iid in enumerate(IDS)
]
FILTERED = "The response was filtered due to the prompt triggering our content management policy."


def _idea(tag, words=20):
    return {
        "problem_gap": f"[Evidence Gap] gap {tag}",
        "core_claim": " ".join(["word"] * words),
        "method": "[Empirical Mapping] method",
        "inspiration": f"i {tag}",
    }


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def _error_400(message):
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=req), body=None)


class FakeAPI:
    """Ideas under an OB card are longer and beat the anchor; the rest lose."""

    def __init__(self, refuse=None):
        self.chat = self.completions = self
        self.models = SimpleNamespace(list=self._models)
        self.requests = []
        self.refuse = refuse or (lambda system, user: False)

    async def _models(self):
        ids = ("muse-spark-1.3-contributor", "muse-wonder-1.0")
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in ids])

    async def create(self, **request):
        self.requests.append(request)
        system, user = (m["content"] for m in request["messages"])
        if self.refuse(system, user):
            raise _error_400(FILTERED)
        if "research scientist" in system:
            k = int(re.search(r"exactly (\d)", user).group(1))
            words = 30 if "oblique constraint" in user else 20
            n = len(self.requests)
            content = json.dumps({"ideas": [_idea(f"{n}-{i}", words) for i in range(k)]})
        elif "expert reviewer" in system:
            claims = [len(s.split()) for s in re.findall(r"Claim: (.*)", user)]
            winner = "A" if len(claims) < 2 or claims[0] >= claims[1] else "B"
            content = json.dumps({"winner": winner})
        elif "field by field" in system:
            content = json.dumps({"problem_gap": 3, "core_claim": 3, "method": 3})
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
def cfg(tmp_path, monkeypatch):
    with open(REAL_CONFIG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    deck = load_deck(DECK_DIR)
    c["decks"]["ML"] = ["OB-001", "OB-004", "OB-005", "OB-007"]
    c["decks"]["MG"] = ["OB-G1-001", "OB-G1-002", "OB-G1-004", "OB-G1-005"]
    assert all(card in deck for d in c["decks"].values() for card in d)
    c |= {
        "cache": str(tmp_path / "cache.db"),
        "draws": str(tmp_path / "draws.json"),
        "wonder": {"ideas_per_condition": 4, "positions": str(tmp_path / "wonder.json")},
        "n_boot_h1": 50,
        "n_boot_other": 50,
    }
    Path(c["draws"]).write_text(json.dumps(draw_cards(IDS, c["decks"])), "utf-8")
    Path(c["wonder"]["positions"]).write_text(json.dumps(draw_wonder_positions(IDS, 4)), "utf-8")
    monkeypatch.setattr(confirm, "split_ids", lambda split: [(i, "p") for i in IDS])
    return c


def runner(cfg, tmp_path, api):
    run = ConfirmB(
        cfg, out=tmp_path / "run", transport=api, wait=wait_none(), embedder=FakeEmbedder()
    )
    run.check_decks = lambda: None
    return run


@pytest.fixture
def unsealed(monkeypatch):
    monkeypatch.setattr(confirm, "load_instances", lambda split: INSTANCES)


def gens(api):
    return [r for r in api.requests if "research scientist" in r["messages"][0]["content"]]


# --- sealed --------------------------------------------------------------------------------


def test_the_plan_never_opens_split_b_nor_calls_the_api(cfg, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("split B was opened")

    monkeypatch.setattr(confirm, "load_instances", refuse)
    monkeypatch.setattr(bench, "read_rows", refuse)
    monkeypatch.delenv(bench.UNSEAL_ENV, raising=False)
    api = FakeAPI()
    plan = runner(cfg, tmp_path, api).plan()
    assert api.requests == []
    assert plan["split_b_opened"] is False
    assert plan["instances"] == 6


def test_a_run_without_the_flag_stops_before_any_call(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv(bench.UNSEAL_ENV, raising=False)
    api = FakeAPI()
    with pytest.raises(SealedSplitError):
        asyncio.run(runner(cfg, tmp_path, api).run())
    assert api.requests == []


def test_the_env_file_does_not_unseal_split_b():
    env = Path(".env")
    if env.exists():
        assert not any(
            line.strip().startswith(bench.UNSEAL_ENV)
            for line in env.read_text("utf-8").splitlines()
        )


# --- the plan ------------------------------------------------------------------------------


def test_the_plan_counts_every_call(cfg, tmp_path):
    plan = runner(cfg, tmp_path, FakeAPI()).plan()
    blocks = {k: v["calls"] for k, v in plan["blocks"].items()}
    n, conds = 6, 7
    assert blocks == {
        "anchor": n,
        "c1": n,
        "sequential": n * 3 * 6,
        "quality": 2 * n * 3 * conds,
        "quality_gold": 2 * n * 3 * conds,
        "quality_wonder": 2 * 4 * conds,
        "grounding": n * 3 * conds,
        "distance": n * 3 * conds,
    }
    assert plan["calls_total"] == sum(blocks.values())
    assert plan["regression"]["data"] == "judgments of split B only"
    assert plan["regression"]["refit_on_every_resample"] is True


def test_the_committed_config_matches_the_amendment_and_the_deck():
    with open(REAL_CONFIG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    assert c["conditions"] == ["C1", "C1s", "Ts", "R", "E6", "ML", "MG"]
    assert (c["n_boot_h1"], c["n_boot_other"]) == (10000, 2000)
    assert c["no_card"]["Ts"] == {"temperature": 1.5}
    assert c["decks"]["ML"] == [f"OB-{i:03d}" for i in (1, 4, 5, 7, 8, 9, 11, 12, 13, 16, 18, 20,
                                                        22, 23, 25, 26, 27, 29)]  # fmt: skip
    assert c["decks"]["MG"] == [f"OB-G1-{i:03d}" for i in (1, 2, 4, 5, 6, 8, 9, 12, 13)]
    deck = load_deck(DECK_DIR)
    assert len(c["decks"]["R"]) == 30 and all(deck[i].origin == "control" for i in c["decks"]["R"])
    assert sorted(c["decks"]["E6"]) == sorted(i for i in deck if deck[i].origin == "paper")
    assert all(deck[i].origin == "seed" for i in c["decks"]["ML"])
    assert all(deck[i].origin == "evolved" for i in c["decks"]["MG"])
    assert "contributor" in c["generator"]["model"]


def test_the_committed_draws_are_the_seeded_draw_of_split_b():
    with open(REAL_CONFIG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    ids = sorted(iid for iid, _ in split_ids("B"))
    assert json.loads(Path(c["draws"]).read_text("utf-8")) == draw_cards(ids, c["decks"])


# --- draws ---------------------------------------------------------------------------------


def test_draws_are_reproducible_and_distinct_inside_a_set():
    decks = {"R": [f"r{i}" for i in range(30)], "E6": list("abcdef"), "ML": list("ghijkl"),
             "MG": list("mnopqr")}  # fmt: skip
    a = draw_cards(IDS, decks)
    assert a == draw_cards(list(reversed(IDS)), {k: v[::-1] for k, v in decks.items()})
    for condition, by_instance in a.items():
        assert set(by_instance) == set(IDS)
        for cards in by_instance.values():
            assert len(set(cards)) == 3 and set(cards) <= set(decks[condition])


def test_changing_one_deck_does_not_change_the_draw_of_another():
    decks = {"R": [f"r{i}" for i in range(30)], "E6": list("abcdef"), "ML": list("ghijkl"),
             "MG": list("mnopqr")}  # fmt: skip
    a = draw_cards(IDS, decks)
    b = draw_cards(IDS, {**decks, "ML": [*decks["ML"], "x", "y"]})
    assert a["R"] == b["R"] and a["E6"] == b["E6"] and a["MG"] == b["MG"]


# --- the verdict ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cu, q, expected",
    [
        ((0.01, 0.1), (-0.04, 0.1), "supported"),
        ((-0.2, -0.01), (-0.04, 0.1), "refuted"),
        ((0.01, 0.1), (-0.3, -0.06), "refuted"),
        ((-0.01, 0.1), (-0.04, 0.1), "inconclusive"),
        ((0.01, 0.1), (-0.06, 0.1), "inconclusive"),
        ((0.01, 0.1), (-0.3, -0.04), "inconclusive"),
    ],
)
def test_h1_verdict_follows_adr_003(cu, q, expected):
    cell = lambda t: {"lo": t[0], "hi": t[1]}  # noqa: E731
    assert h1_verdict(cell(cu), cell(q), 0.05) == expected


# --- a run ---------------------------------------------------------------------------------


def test_a_run_generates_each_condition_as_registered(cfg, tmp_path, unsealed):
    api = FakeAPI()
    run = runner(cfg, tmp_path, api)
    asyncio.run(run.run())
    n = len(IDS)
    calls = gens(api)
    assert len(calls) == n + n + n * 3 * 6  # anchors, C1, sequences
    assert sum("exactly 3" in r["messages"][1]["content"] for r in calls) == n
    hot = [r for r in calls if r["temperature"] == 1.5]
    assert len(hot) == n * 3
    assert all("oblique constraint" not in r["messages"][1]["content"] for r in hot)
    assert sum("oblique constraint" in r["messages"][1]["content"] for r in calls) == n * 3 * 4
    deck = load_deck(DECK_DIR)
    draws = run.expected_draws()
    for x in IDS:
        records = run.slots[(x, "ML")]
        assert [r.card for r in records] == draws["ML"][x]
        assert [r.slot for r in records] == [0, 1, 2]
        assert "+" in records[0].prompt_hash
        for card in draws["ML"][x]:
            assert any(deck[card].text in r["messages"][1]["content"] for r in calls)
        assert all(r.card is None for r in run.slots[(x, "C1s")])


def test_later_ideas_of_a_set_see_the_earlier_ones(cfg, tmp_path, unsealed):
    api = FakeAPI()
    run = runner(cfg, tmp_path, api)
    asyncio.run(run.run())
    first, second = run.slots[(IDS[0], "E6")][:2]
    assert first.idea.problem_gap != second.idea.problem_gap
    with_previous = [r for r in gens(api) if first.idea.problem_gap in r["messages"][1]["content"]]
    assert with_previous


def test_the_regression_uses_only_this_run_and_every_order(cfg, tmp_path, unsealed):
    run = runner(cfg, tmp_path, FakeAPI())
    asyncio.run(run.run())
    rows = run.regression_rows()
    assert len(rows) == 2 * len(IDS) * 3 * 7
    assert {r["instance"] for r in rows} == set(IDS)
    assert {r["condition"] for r in rows} == set(cfg["conditions"])


def test_the_report_opens_with_h1_and_the_guard_before_the_exploratory(cfg, tmp_path, unsealed):
    run = runner(cfg, tmp_path, FakeAPI())
    asyncio.run(run.run())
    result = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))
    order = list(result)
    assert order.index("h1") < order.index("grounding_guard") < order.index("exploratory")
    assert result["h1"]["verdict"] in ("supported", "refuted", "inconclusive")
    assert "bca_lo" in result["h1"]["delta_cu_ml_c1s"]
    assert "bca_lo" in result["h1"]["delta_q_controlled_ml_c1s"]
    ex = result["exploratory"]
    for key in ("h3_cu_ml_ts", "h3_cu_ml_r", "h4_cu_mg_ml", "h6_cu_e6_c1s", "h7_cu_e6_ml"):
        assert ex[key] is not None
    assert set(ex["delta_grounding_vs_c1s"]) == {"E6", "ML", "MG", "R"}
    assert set(result["conditions"]) == set(cfg["conditions"])
    text = (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert text.index("## H1") < text.index("## Grounding guard") < text.index("## Exploratory")


def test_q_against_the_gold_follows_the_config(cfg, tmp_path, unsealed):
    api = FakeAPI()
    cfg["q_gold"] = False
    asyncio.run(runner(cfg, tmp_path, api).run())
    assert not any("Gold." in r["messages"][1]["content"] for r in api.requests)
    plan = runner(cfg, tmp_path, FakeAPI()).plan()
    assert plan["blocks"]["quality_gold"]["calls"] == 0


# --- Wonder --------------------------------------------------------------------------------


def test_wonder_judges_the_same_positions_in_every_condition_against_the_anchor(
    cfg, tmp_path, unsealed
):
    api = FakeAPI()
    run = runner(cfg, tmp_path, api)
    asyncio.run(run.run())
    calls = [r for r in api.requests if r["model"] == "muse-wonder-1.0"]
    assert len(calls) == 2 * 4 * 7
    assert all(r["temperature"] == 0 for r in calls)
    assert not any("Gold." in r["messages"][1]["content"] for r in calls)
    positions = {tuple(p) for p in draw_wonder_positions(IDS, 4)}
    for cond in cfg["conditions"]:
        judged = {(i, s) for i, c, s in run.q["wonder"] if c == cond}
        assert judged == positions
    orders = [r["messages"][1]["content"] for r in calls]
    assert len(set(orders)) == len(orders)
    wonder = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))["sensitivity"][
        "wonder"
    ]
    assert wonder["ideas_judged"] == 4 * 7
    assert wonder["agreement_q"]["n"] == 4 * 7 and wonder["agreement_per_order"]["n"] == 2 * 4 * 7
    assert wonder["agreement_q"]["exact"] == 1.0  # the fake judges by the same rule
    assert wonder["delta_q_ml_c1s_wonder"] == wonder["delta_q_ml_c1s_spark_same_ideas"]
    assert "Wonder on 28 ideas" in (tmp_path / "run" / "summary.md").read_text("utf-8")


def test_wonder_does_not_enter_the_regression(cfg, tmp_path, unsealed):
    run = runner(cfg, tmp_path, FakeAPI())
    asyncio.run(run.run())
    assert len(run.regression_rows()) == 2 * len(IDS) * 3 * 7


def test_a_stored_wonder_draw_that_differs_stops_the_run(cfg, tmp_path, unsealed):
    Path(cfg["wonder"]["positions"]).write_text(json.dumps([["i-1000", 0]]), "utf-8")
    api = FakeAPI()
    with pytest.raises(SystemExit, match="differs from the seeded draw"):
        asyncio.run(runner(cfg, tmp_path, api).run())
    assert api.requests == []


def test_the_committed_wonder_positions_are_the_seeded_draw_of_split_b():
    with open(REAL_CONFIG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    ids = sorted(iid for iid, _ in split_ids("B"))
    assert c["wonder"]["ideas_per_condition"] == 30
    stored = json.loads(Path(c["wonder"]["positions"]).read_text("utf-8"))
    assert stored == draw_wonder_positions(ids, 30)
    assert len({tuple(p) for p in stored}) == 30
    assert (c["budget"]["approved_usd"], c["budget"]["stop_usd"]) == (4.0, 3.7)
    assert c["q_gold"] is True and "C1" in c["conditions"]


def test_the_generator_timeout_comes_from_the_config_and_not_the_cache_key(cfg, tmp_path):
    from oblique.cache import make_key

    run = runner(cfg, tmp_path, FakeAPI())
    seen = []
    original = run.new_client

    def spy(provider, concurrency=1, **kw):
        seen.append(kw.get("timeout"))
        return original(provider, concurrency, **kw)

    run.new_client = spy
    cfg["generator"]["timeout_seconds"] = 900
    run.check_draws = lambda: run.expected_draws()
    run.generate_anchors = lambda client, instances: asyncio.sleep(0)
    asyncio.run(run.generate([]))
    assert seen == [900]
    messages = [{"role": "user", "content": "x"}]
    assert make_key("m", messages, {"a": 1}, 0, 0, provider="meta") == make_key(
        "m", messages, {"a": 1}, 0, 0, provider="meta"
    )


# --- provider errors -----------------------------------------------------------------------


def test_a_provider_error_is_not_asked_and_its_instance_leaves_only_that_condition(
    cfg, tmp_path, unsealed
):
    cfg["provider_errors"] = [{"instance": IDS[0], "condition": "Ts", "slot": 2}]
    api = FakeAPI()
    run = runner(cfg, tmp_path, api)
    asyncio.run(run.run())
    assert len(gens(api)) == len(IDS) * 2 + len(IDS) * 3 * 6 - 1
    record = run.slots[(IDS[0], "Ts")][2]
    assert record.invalid_reason == "provider_error" and not record.valid
    result = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))
    assert result["conditions"]["Ts"]["sets_complete"] == len(IDS) - 1
    assert result["conditions"]["Ts"]["provider_error"] == 1
    assert result["conditions"]["Ts"]["invalid"] == 0
    assert result["conditions"]["ML"]["sets_complete"] == len(IDS)
    assert result["exploratory"]["h3_cu_ml_ts"]["n"] == len(IDS) - 1
    assert result["exploratory"]["h3_cu_ml_r"]["n"] == len(IDS)
    assert result["h1"]["delta_cu_ml_c1s"]["n"] == len(IDS)
    sensitivity = result["sensitivity"]["provider_error_as_invalid"]
    assert sensitivity["h3_cu_ml_ts"]["n"] == len(IDS)
    assert result["provider_errors"] == [{"instance": IDS[0], "condition": "Ts", "slot": 2}]
    assert len(run.regression_rows()) == 2 * (len(IDS) * 3 * 7 - 1)
    text = (tmp_path / "run" / "summary.md").read_text("utf-8")
    assert "provider error as q = 0" in text and f"(n = {len(IDS) - 1})" in text


def test_without_provider_errors_there_is_no_such_sensitivity(cfg, tmp_path, unsealed):
    asyncio.run(runner(cfg, tmp_path, FakeAPI()).run())
    result = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))
    assert result["sensitivity"]["provider_error_as_invalid"] is None
    assert result["provider_errors"] == []


def test_the_committed_config_lists_the_provider_error_of_split_b():
    with open(REAL_CONFIG, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    assert c["provider_errors"] == [{"instance": "i-4183", "condition": "Ts", "slot": 2}]


# --- once ----------------------------------------------------------------------------------


def test_a_finished_run_refuses_to_start_again(cfg, tmp_path, unsealed):
    asyncio.run(runner(cfg, tmp_path, FakeAPI()).run())
    api = FakeAPI()
    with pytest.raises(SystemExit, match="runs once"):
        asyncio.run(runner(cfg, tmp_path, api).run())
    assert api.requests == []


def test_a_run_that_was_cut_resumes_and_pays_only_for_what_is_missing(cfg, tmp_path, unsealed):
    first = FakeAPI()

    class Cut(Exception):
        pass

    async def cut(self, instances):
        raise Cut

    original = ConfirmB.judge
    ConfirmB.judge = cut
    try:
        with pytest.raises(Cut):
            asyncio.run(runner(cfg, tmp_path, first).run())
    finally:
        ConfirmB.judge = original
    paid = len(first.requests)
    assert paid == len(gens(first))

    second = FakeAPI()
    run = runner(cfg, tmp_path, second)
    asyncio.run(run.run())
    assert gens(second) == []
    assert run.spent_usd > 0


def test_a_call_the_filter_refused_is_not_asked_again_on_resume(cfg, tmp_path, unsealed):
    refused = lambda system, user: "expert reviewer" in system  # noqa: E731
    first = FakeAPI(refuse=refused)

    original = ConfirmB.embed
    ConfirmB.embed = lambda self: (_ for _ in ()).throw(RuntimeError("cut"))
    try:
        with pytest.raises(RuntimeError, match="cut"):
            asyncio.run(runner(cfg, tmp_path, first).run())
    finally:
        ConfirmB.embed = original
    judged = sum("expert reviewer" in r["messages"][0]["content"] for r in first.requests)
    assert judged > 0

    second = FakeAPI()
    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(runner(cfg, tmp_path, second).run())
    assert not any("expert reviewer" in r["messages"][0]["content"] for r in second.requests)


def test_the_client_remembers_a_filtered_key(tmp_path):
    log = tmp_path / "filtered.jsonl"
    api = FakeAPI(refuse=lambda system, user: True)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    async def ask(client):
        return await client.complete("m", messages)

    client = LLMClient("meta", Cache(tmp_path / "c.db"), client=api, filtered_log=log)
    with pytest.raises(ContentFilteredError):
        asyncio.run(ask(client))
    assert len(api.requests) == 2
    again = LLMClient("meta", Cache(tmp_path / "c.db"), client=api, filtered_log=log)
    with pytest.raises(ContentFilteredError):
        asyncio.run(ask(again))
    assert len(api.requests) == 2


# --- BCa -----------------------------------------------------------------------------------


def test_bca_matches_percentile_on_symmetric_data():
    values = np.linspace(-1, 1, 41)
    percentile = bootstrap_mean_ci(values, n_boot=4000, seed=1)
    check = bca_mean_ci(values, n_boot=4000, seed=1)
    assert check.mean == pytest.approx(percentile.mean)
    assert check.lo == pytest.approx(percentile.lo, abs=0.02)
    assert check.hi == pytest.approx(percentile.hi, abs=0.02)


def test_bca_shifts_toward_the_long_tail_on_skewed_data():
    values = np.random.default_rng(3).exponential(1.0, 50)
    percentile = bootstrap_mean_ci(values, n_boot=4000, seed=1)
    check = bca_mean_ci(values, n_boot=4000, seed=1)
    assert check.lo > percentile.lo and check.hi > percentile.hi


def test_bca_of_constant_values_is_the_value():
    assert bca_interval(0.5, np.full(100, 0.5), np.full(10, 0.5)) == (0.5, 0.5)


def test_the_regression_can_add_a_bca_check_without_changing_its_interval():
    gen = np.random.default_rng(5)
    rows = [
        {
            "instance": f"i{i}",
            "condition": c,
            "win": bool(gen.random() < (0.6 if c == "a" else 0.4)),
            "delta_words": int(gen.integers(-5, 6)),
        }
        for i in range(30)
        for c in ("a", "b")
        for _ in range(2)
    ]
    plain = controlled_win_rates(rows, ["a", "b"], comparisons=[("a", "b")], n_boot=200, seed=1)
    checked = controlled_win_rates(
        rows, ["a", "b"], comparisons=[("a", "b")], n_boot=200, seed=1, bca=True
    )
    a, b = plain["differences"]["a-b"], checked["differences"]["a-b"]
    assert (a["mean"], a["lo"], a["hi"]) == (b["mean"], b["lo"], b["hi"])
    assert b["bca_lo"] < b["bca_hi"]
    assert "bca_lo" not in a


def test_main_rejects_unknown_flags():
    with pytest.raises(SystemExit):
        main([REAL_CONFIG, "--go"])
