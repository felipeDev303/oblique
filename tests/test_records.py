import json

import pytest

from oblique.generate.labels import LabelStatus
from oblique.generate.records import read_generations, records_from_set, write_jsonl
from oblique.generate.validate import InvalidReason, parse_set
from oblique.llm import Usage


def idea(gap: str, method: str) -> dict:
    return {"problem_gap": gap, "core_claim": "c", "method": method, "inspiration": "i"}


SET = [
    idea("[Evidence Gap] g", "[Empirical Mapping] m"),  # ok / ok
    idea("[Other] g", "[Made Up Paradigm] m"),  # other / off_list
    idea("no label g", "[robustification] m"),  # missing / ok
]
USAGE = Usage(tokens_in=1000, tokens_cached=800, tokens_out=300, tokens_reasoning=50)


def build(text: str, k: int = 3):
    return records_from_set(
        parse_set(text, k),
        cards=["OB-001", "OB-002", "OB-003"][:k],
        run_id="R-20261030-01",
        instance="i-0042",
        condition="M0",
        model="muse-spark",
        prompt_hash="abc123",
        seed=0,
        call_key="k" * 64,
        usage=USAGE,
    )


def test_label_status_per_axis():
    records = build(json.dumps(SET))
    assert [r.opportunity_label_status for r in records] == [
        LabelStatus.ok,
        LabelStatus.other,
        LabelStatus.missing,
    ]
    assert [r.method_label_status for r in records] == [
        LabelStatus.ok,
        LabelStatus.off_list,
        LabelStatus.ok,
    ]
    assert records[0].opportunity_label == "Evidence Gap"
    assert records[1].method_label is None
    assert records[1].method_label_raw == "Made Up Paradigm"
    assert records[2].method_label == "Robustification"


def test_conforming_requires_both_axes():
    assert [r.conforming for r in build(json.dumps(SET))] == [True, False, False]


def test_every_line_carries_run_conventions():
    for i, r in enumerate(build(json.dumps(SET))):
        assert (r.run_id, r.instance, r.card, r.slot) == (
            "R-20261030-01",
            "i-0042",
            f"OB-00{i + 1}",
            i,
        )
        assert (r.model, r.prompt_hash, r.seed) == ("muse-spark", "abc123", 0)
        assert (r.tokens_in, r.tokens_out) == (1000, 300)


def test_invalid_ideas_have_no_label_status():
    records = build("not json")
    assert all(not r.valid for r in records)
    assert all(r.invalid_reason is InvalidReason.json_error for r in records)
    assert all(
        r.opportunity_label_status is None and r.method_label_status is None for r in records
    )
    assert not any(r.conforming for r in records)


def test_fenced_and_shape_are_recorded():
    plain = build(json.dumps(SET))
    assert all(not r.fenced and r.shape == "list" for r in plain)
    fenced = build("```json\n" + json.dumps({"ideas": SET}) + "\n```")
    assert all(r.fenced and r.shape == "ideas_object" for r in fenced)
    assert all(r.valid for r in fenced)


def test_card_list_must_match_k():
    with pytest.raises(ValueError):
        records_from_set(
            parse_set(json.dumps(SET), 3),
            cards=["OB-001"],
            run_id="r",
            instance="i",
            condition="M0",
            model="m",
            prompt_hash="h",
            seed=0,
            call_key="k",
            usage=USAGE,
        )


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "runs" / "R-20261030-01" / "generations.jsonl"
    records = build(json.dumps(SET))
    write_jsonl(path, records)
    write_jsonl(path, build("not json"), append=True)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    assert first["opportunity_label_status"] == "ok"
    assert first["method_label_status"] == "ok"
    assert json.loads(lines[1])["method_label_status"] == "off_list"
    assert json.loads(lines[3])["opportunity_label_status"] is None

    assert read_generations(path)[:3] == records
