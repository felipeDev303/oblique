import pytest

from oblique.generate.labels import (
    METHOD_PARADIGMS,
    OPPORTUNITY_PATTERNS,
    LabelStatus,
    classify_method,
    classify_opportunity,
    normalize_label,
)


def test_eight_labels_per_axis_including_other():
    for labels in (OPPORTUNITY_PATTERNS, METHOD_PARADIGMS):
        assert len(labels) == 8
        assert len({normalize_label(x) for x in labels}) == 8
        assert labels[-1] == "Other"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Evidence Gap", "evidence gap"),
        ("EVIDENCE GAP", "evidence gap"),
        ("  Evidence    Gap ", "evidence gap"),
        ("Evidence\tGap", "evidence gap"),
        ("Puzzle / Contradiction", "puzzle/contradiction"),
        ("Puzzle/Contradiction", "puzzle/contradiction"),
        ("Puzzle  /   Contradiction", "puzzle/contradiction"),
        ("puzzle /contradiction", "puzzle/contradiction"),
    ],
)
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize(
    "text, canonical",
    [
        ("[Evidence Gap] the field lacks data", "Evidence Gap"),
        ("[evidence gap] lower case", "Evidence Gap"),
        ("  [Evidence  Gap]leading spaces, no space after", "Evidence Gap"),
        ("[Puzzle/Contradiction] no spaces around slash", "Puzzle / Contradiction"),
        ("[FAILURE /RISK GAP] mixed", "Failure / Risk Gap"),
    ],
)
def test_canonical_opportunity_labels(text, canonical):
    result = classify_opportunity(text)
    assert result.status is LabelStatus.ok
    assert result.canonical == canonical
    assert result.conforming


def test_method_axis_uses_its_own_list():
    assert classify_method("[Relax/Extend Scope] x").canonical == "Relax / Extend Scope"
    # a valid opportunity label is off-list on the method axis
    assert classify_method("[Evidence Gap] x").status is LabelStatus.off_list


@pytest.mark.parametrize("text", ["[Other] x", "[other] x", "[ OTHER ] x"])
def test_other(text):
    result = classify_opportunity(text)
    assert result.status is LabelStatus.other
    assert result.canonical == "Other"
    assert result.conforming


def test_off_list_keeps_raw_label():
    result = classify_opportunity("[Novel Mechanism] something new")
    assert result.status is LabelStatus.off_list
    assert result.canonical is None
    assert result.raw == "Novel Mechanism"
    assert not result.conforming


@pytest.mark.parametrize(
    "text",
    [
        "Evidence Gap: no brackets",
        "Evidence Gap the field lacks data",
        "(Evidence Gap) wrong brackets",
        "The gap is [Evidence Gap] mid-sentence",
        "[] empty brackets",
        "[   ] blank brackets",
        "",
    ],
)
def test_missing(text):
    result = classify_opportunity(text)
    assert result.status is LabelStatus.missing
    assert result.canonical is None
    assert not result.conforming


def test_only_the_leading_label_counts():
    result = classify_opportunity("[Evidence Gap] [Scope Mismatch] two labels")
    assert result.canonical == "Evidence Gap"
