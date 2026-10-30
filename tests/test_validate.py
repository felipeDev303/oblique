import json

import pytest

from oblique.generate.validate import InvalidReason, parse_set


def idea(**overrides) -> dict:
    base = {
        "problem_gap": "[Evidence Gap] gap text",
        "core_claim": "claim text",
        "method": "[Empirical Mapping] method text",
        "inspiration": "inspiration text",
    }
    return {**base, **overrides}


def reasons(parsed) -> list:
    return [s.reason for s in parsed.slots]


def test_valid_array():
    parsed = parse_set(json.dumps([idea(), idea(), idea()]), k=3)
    assert len(parsed.valid) == 3
    assert parsed.invalid == []
    assert not parsed.retryable


def test_valid_object_with_ideas_key():
    parsed = parse_set(json.dumps({"ideas": [idea(), idea(), idea()]}), k=3)
    assert len(parsed.valid) == 3


def test_surrounding_whitespace_is_fine():
    assert len(parse_set("\n  " + json.dumps([idea()]) + "\n", k=1).valid) == 1


def test_labels_do_not_affect_validity():
    parsed = parse_set(
        json.dumps([idea(problem_gap="no label here", method="[Made Up] text")]), k=1
    )
    assert len(parsed.valid) == 1


def test_text_is_kept_verbatim():
    parsed = parse_set(json.dumps([idea(core_claim="  odd   spacing ")]), k=1)
    assert parsed.valid[0].core_claim == "  odd   spacing "


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '[{"problem_gap": "x"',
        "```json\n[]\n```",  # code fences are not JSON
        "",
        None,
    ],
)
def test_unparseable_json_invalidates_all_and_is_retryable(text):
    parsed = parse_set(text, k=3)
    assert parsed.valid == []
    assert reasons(parsed) == [InvalidReason.json_error] * 3
    assert parsed.retryable


@pytest.mark.parametrize("n", [0, 2, 4])
def test_wrong_count_invalidates_all(n):
    parsed = parse_set(json.dumps([idea()] * n), k=3)
    assert reasons(parsed) == [InvalidReason.wrong_count] * 3
    assert not parsed.retryable


@pytest.mark.parametrize("payload", [idea(), "a string", 42, {"other": []}])
def test_wrong_shape_invalidates_all(payload):
    parsed = parse_set(json.dumps(payload), k=1)
    assert reasons(parsed) == [InvalidReason.bad_shape]
    assert not parsed.retryable


def test_missing_field_invalidates_only_that_idea():
    broken = idea()
    del broken["inspiration"]
    parsed = parse_set(json.dumps([idea(), broken, idea()]), k=3)
    assert reasons(parsed) == [None, InvalidReason.missing_field, None]
    assert parsed.slots[1].detail == "inspiration"
    assert len(parsed.valid) == 2


@pytest.mark.parametrize("blank", ["", " ", "\n\t "])
def test_empty_or_whitespace_field_is_invalid(blank):
    parsed = parse_set(json.dumps([idea(method=blank)]), k=1)
    assert reasons(parsed) == [InvalidReason.empty_field]
    assert parsed.slots[0].detail == "method"


@pytest.mark.parametrize("value", [None, 3, ["a"], {"a": 1}])
def test_non_text_field_is_invalid(value):
    parsed = parse_set(json.dumps([idea(core_claim=value)]), k=1)
    assert reasons(parsed) == [InvalidReason.not_text]


def test_non_object_element_is_invalid():
    parsed = parse_set(json.dumps([idea(), "just text"]), k=2)
    assert reasons(parsed) == [None, InvalidReason.not_object]


def test_slots_keep_their_position():
    parsed = parse_set(json.dumps([idea(method=""), idea(), idea(core_claim="")]), k=3)
    assert [s.index for s in parsed.slots] == [0, 1, 2]
    assert [s.valid for s in parsed.slots] == [False, True, False]


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        parse_set("[]", k=0)
