import json

import pytest

from oblique.generate.validate import InvalidReason, filtered_set, parse_set


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
        "",
        None,
    ],
)
def test_unparseable_json_invalidates_all_and_is_retryable(text):
    parsed = parse_set(text, k=3)
    assert parsed.valid == []
    assert reasons(parsed) == [InvalidReason.json_error] * 3
    assert parsed.retryable
    assert not parsed.fenced


# --- Code fences (amendment 2) ----------------------------------------------

THREE = json.dumps([idea(), idea(), idea()], indent=2)


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{THREE}\n```",
        f"```\n{THREE}\n```",
        f"  \n```json\n{THREE}\n```\n\n",  # whitespace around the block is fine
        f"```json  \r\n{THREE}\r\n```",
        f"```json\n{THREE}```",  # closing fence right after the JSON
    ],
)
def test_single_whole_response_fence_is_unwrapped(text):
    parsed = parse_set(text, k=3)
    assert len(parsed.valid) == 3
    assert parsed.fenced


def test_unfenced_response_is_not_marked_fenced():
    assert not parse_set(THREE, k=3).fenced


@pytest.mark.parametrize(
    "text",
    [
        f"Here are the ideas:\n```json\n{THREE}\n```",  # text before
        f"```json\n{THREE}\n```\nHope this helps!",  # text after
        f"```json\n{THREE}\n```\n```json\n{THREE}\n```",  # two blocks
        f"```python\n{THREE}\n```",  # other language tag
        f"```json {THREE}```",  # no newline after the opening fence
    ],
)
def test_anything_else_around_the_json_is_json_error(text):
    parsed = parse_set(text, k=3)
    assert reasons(parsed) == [InvalidReason.json_error] * 3
    assert parsed.retryable


def test_fenced_invalid_set_keeps_fenced_flag():
    parsed = parse_set("```json\n[]\n```", k=3)
    assert reasons(parsed) == [InvalidReason.wrong_count] * 3
    assert parsed.fenced


# --- Shape (amendment 4) ----------------------------------------------------


def test_shape_is_recorded():
    assert parse_set(json.dumps([idea()]), k=1).shape == "list"
    assert parse_set(json.dumps({"ideas": [idea()]}), k=1).shape == "ideas_object"
    assert parse_set(json.dumps({"ideas": [idea()] * 2}), k=1).shape == "ideas_object"
    assert parse_set("nope", k=1).shape is None
    assert parse_set(json.dumps({"other": []}), k=1).shape is None


def test_ideas_key_must_hold_a_list():
    parsed = parse_set(json.dumps({"ideas": "text"}), k=1)
    assert reasons(parsed) == [InvalidReason.bad_shape]


# --- Retry seed (amendment 3) -----------------------------------------------


def test_api_seed_never_collides_with_another_sample():
    from oblique.generate.validate import api_seed

    seeds = {api_seed(s, a) for s in range(3) for a in range(2)}
    assert len(seeds) == 6
    assert api_seed(0, 0) == 0
    assert api_seed(2, 1) == 1002
    with pytest.raises(ValueError):
        api_seed(1000, 0)


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


def test_filtered_set_is_invalid_and_not_retryable():
    parsed = filtered_set(3)
    assert parsed.k == 3 and parsed.valid == [] and not parsed.retryable
    assert {s.reason for s in parsed.invalid} == {InvalidReason.content_filtered}
