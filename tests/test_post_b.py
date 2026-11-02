"""The analyses after split B, from the records of a fake run. Nothing here opens split B."""

import asyncio
import json

import pytest
import test_confirm
from test_confirm import IDS, FakeAPI, FakeEmbedder, runner

from oblique import confirm
from oblique.post_b import SECTION, PostB, append_section, final_values, product_deck

cfg = test_confirm.cfg  # the fixtures of the confirmation runner
unsealed = test_confirm.unsealed


def finished_run(cfg, tmp_path, api=None):
    asyncio.run(runner(cfg, tmp_path, api or FakeAPI()).run())
    return PostB(cfg, out=tmp_path / "run", embedder=FakeEmbedder())


def sealed(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("split B was opened")

    monkeypatch.setattr(confirm, "load_instances", refuse)


def test_the_rebuilt_state_gives_the_numbers_of_the_run_with_no_call(
    cfg, tmp_path, unsealed, monkeypatch
):
    cfg["provider_errors"] = [{"instance": IDS[0], "condition": "Ts", "slot": 2}]
    post = finished_run(cfg, tmp_path)
    sealed(monkeypatch)
    post.new_client = lambda *a, **k: pytest.fail("a client was made")
    post.load()
    post.embed()
    result = post.analyses()
    assert result["rebuilt_matches_run"]
    assert result["q_controlled_mg_c1s"]["n"] == len(IDS)
    assert set(result["c1s_vs_c1"]) == {"cu", "q_controlled", "q_gold", "d", "d_embed"}
    assert set(result["d_vs_c1s"]) == {"R", "E6", "ML", "MG"}
    assert result["product"]["deck"] in ("ML", "MG", "E6")


def test_a_filtered_judgment_rebuilds_as_missing(cfg, tmp_path, unsealed, monkeypatch):
    first = []

    def refused(system, user):
        if "expert reviewer" not in system:
            return False
        first[:] = first or [user]
        return user == first[0]

    post = finished_run(cfg, tmp_path, FakeAPI(refuse=refused))
    sealed(monkeypatch)
    post.load()
    post.embed()
    assert post.content_filtered
    assert post.analyses()["rebuilt_matches_run"]
    assert None in post.q["anchor"].values() or None in post.q["spark"].values()


def test_a_changed_record_stops_the_analysis(cfg, tmp_path, unsealed):
    post = finished_run(cfg, tmp_path)
    path = tmp_path / "run" / "judgments.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    flipped = next(r for r in rows if r["judge"] == "quality-anchor")
    flipped["parsed"] = "B" if flipped["parsed"] == "A" else "A"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    post.load()
    post.embed()
    with pytest.raises(SystemExit, match="differ"):
        post.analyses()


def test_the_last_attempt_of_a_judge_call_is_its_value():
    rows = [
        {"judge": "grounding-spark", "instance": "i", "condition": "C1", "slot": 0,
         "attempt": 0, "parsed": None, "parse_error": "bad"},
        {"judge": "grounding-spark", "instance": "i", "condition": "C1", "slot": 0,
         "attempt": 1, "parsed": 4, "parse_error": None},
        {"judge": "quality-anchor", "instance": "i", "condition": "C1", "slot": 0,
         "idea_first": True, "attempt": 0, "parsed": None, "parse_error": "content_filter"},
    ]  # fmt: skip
    values = final_values(rows)
    assert values[("grounding-spark", "i", "C1", 0)] == 4
    assert values[("quality-anchor", "i", "C1", 0, True)] is None


def _cell(mean, lo=None):
    return {"mean": mean, "lo": mean - 0.1 if lo is None else lo, "hi": mean + 0.1, "n": 49}


def test_the_product_deck_is_the_highest_cu_among_those_that_pass():
    q = {"ML": _cell(0.0, -0.04), "MG": _cell(0.0, -0.03), "E6": _cell(0.1, -0.08)}
    cu = {"ML": _cell(0.05), "MG": _cell(0.02), "E6": _cell(0.09)}
    grounding = {"ML": _cell(-0.05), "MG": _cell(-0.05), "E6": _cell(-0.05)}
    out = product_deck(q, cu, grounding, 0.05, -0.10)
    assert out == out | {"deck": "ML", "experimental": False, "passing": ["ML", "MG"]}


def test_the_guard_is_on_the_mean_and_with_none_passing_the_rule_takes_the_highest_q():
    q = {"ML": _cell(0.01, -0.04), "MG": _cell(0.02, -0.06), "E6": _cell(0.03, -0.06)}
    cu = {"ML": _cell(0.05), "MG": _cell(0.02), "E6": _cell(0.09)}
    grounding = {"ML": _cell(-0.105, -0.0), "MG": _cell(-0.05, -0.2), "E6": _cell(-0.05)}
    out = product_deck(q, cu, grounding, 0.05, -0.10)
    assert out["checks"]["ML"]["guard"] is False and out["checks"]["MG"]["guard"] is True
    assert out == out | {"deck": "E6", "experimental": True, "passing": []}


def test_the_section_replaces_an_earlier_one(tmp_path):
    summary = tmp_path / "summary.md"
    summary.write_text("# Run\n\nbody\n", "utf-8")
    append_section(summary, f"{SECTION}\n\nfirst\n")
    append_section(summary, f"{SECTION}\n\nsecond\n")
    text = summary.read_text("utf-8")
    assert text.count(SECTION) == 1 and "second" in text and "first" not in text
    assert text.startswith("# Run\n\nbody\n\n")
