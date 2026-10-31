import pytest

from oblique.bench import Instance
from oblique.generate.labels import METHOD_PARADIGMS, OPPORTUNITY_PATTERNS, strip_leading_label
from oblique.generate.schema import IdeaCard
from oblique.prompts import (
    IDEATION_V3,
    PROMPTS,
    JudgeOutputError,
    Prompt,
    distance,
    distance_messages,
    field_words,
    grounding,
    grounding_messages,
    ideation_messages,
    judged_words,
    parse_grounding,
    parse_similarity,
    parse_winner,
    q_from_orders,
    quality_messages,
    render_card,
)
from oblique.prompts.ideation import METHOD_DEFINITIONS, OPPORTUNITY_DEFINITIONS

PAPERS = Instance(
    id="i-0001",
    pair_id="p",
    arxiv_id="2510.00001",
    domain="ML/AI",
    paper1_title="Paper One",
    paper2_title="Paper Two",
    abstract_1="First summary with {braces}.",
    abstract_2="Second summary.",
    gold="Gold insight paragraph.",
)

IDEA = IdeaCard(
    problem_gap="[Evidence Gap] Nobody has measured X.",
    core_claim="X predicts Y.",
    method="[Empirical Mapping]: Measure X across Z.",
    inspiration="Paper 1's X and Paper 2's Y.",
)

# Changing a template changes its hash: bump the version (ideation-v4, ...) and update here.
FROZEN = {
    "ideation-v2": "cbadaa8eb5837db2",
    "ideation-v3": "ac302619d59086d2",
    "quality-v1": "a6faec078dabf17b",
    "quality-v2": "6bdf589b7106e1a4",
    "distance-v1": "28a29c13b725e007",
    "grounding-v1": "b8fdc3ca41e79408",
}


def test_registry_ids_and_hashes():
    assert set(PROMPTS) == set(FROZEN)
    assert all(PROMPTS[i].id == i for i in PROMPTS)
    assert len({p.hash for p in PROMPTS.values()}) == len(PROMPTS)
    assert {i: p.hash for i, p in PROMPTS.items()} == FROZEN


def test_ideation_defines_every_canonical_label_in_one_line():
    assert tuple(OPPORTUNITY_DEFINITIONS) == OPPORTUNITY_PATTERNS
    assert tuple(METHOD_DEFINITIONS) == METHOD_PARADIGMS
    system = ideation_messages(PAPERS, 3)[0]["content"]
    lines = system.splitlines()
    for definitions in (OPPORTUNITY_DEFINITIONS, METHOD_DEFINITIONS):
        for label, text in definitions.items():
            assert text and "\n" not in text
            assert f"- [{label}]: {text}" in lines
    assert '{"ideas": [ ... ]}' in system
    for field in ("problem_gap", "core_claim", "method", "inspiration"):
        assert f'"{field}"' in system


def test_ideation_user_message_puts_papers_first():
    user = ideation_messages(PAPERS, 3)[1]["content"]
    assert user.startswith("<papers>")
    assert "First summary with {braces}." in user
    assert "Propose 3 distinct research ideas based on" in user and "exactly 3 objects" in user
    one = ideation_messages(PAPERS, 1)[1]["content"]
    assert "Propose 1 distinct research idea based on" in one and "exactly 1 object." in one


def test_ideation_calls_share_the_prefix():
    a, b = ideation_messages(PAPERS, 1), ideation_messages(PAPERS, 3)
    assert a[0] == b[0]
    prefix = a[1]["content"].split("</papers>")[0]
    assert b[1]["content"].startswith(prefix)


def test_v3_replaces_the_sentence_rule_with_word_limits():
    system = IDEATION_V3.system
    assert "one to three sentences" not in system
    limits = ('"problem_gap" 40', '"core_claim" 40', '"method" 60', '"inspiration" 80')
    assert all(limit in system for limit in limits)
    v2 = PROMPTS["ideation-v2"].system
    assert "one to three sentences" in v2 and "Word limits" not in v2


def test_first_sequential_idea_reads_like_any_k1_call():
    assert ideation_messages(PAPERS, 1, previous=()) == ideation_messages(PAPERS, 1)


def test_sequential_text_is_fixed_and_carries_the_ideas_as_generated():
    user = ideation_messages(PAPERS, 1, previous=[IDEA, IDEA])[1]["content"]
    assert user.startswith("<papers>")
    assert "Ideas already proposed from these papers:" in user
    assert "Idea 1:\nproblem_gap: [Evidence Gap] Nobody has measured X.\ncore_claim: X" in user
    assert "Idea 2:" in user and "inspiration: Paper 1's X and Paper 2's Y." in user
    assert user.endswith(
        "The new idea must be distinct from these.\n\nPropose 1 distinct research idea based on "
        'these papers. The "ideas" array must contain exactly 1 object.'
    )


def test_sequential_generation_needs_k1_and_a_slot():
    with pytest.raises(ValueError):
        ideation_messages(PAPERS, 3, previous=[IDEA])
    with pytest.raises(ValueError):
        ideation_messages(PAPERS, 1, PROMPTS["ideation-v2"], previous=[IDEA])


def test_extra_blocks_enter_the_hash():
    a = Prompt("x-v1", "s", "u", extra=(("block", "one"),))
    b = Prompt("x-v1", "s", "u", extra=(("block", "two"),))
    assert a.hash != b.hash != Prompt("x-v1", "s", "u").hash


def test_quality_v2_adds_only_the_length_line():
    v1, v2 = PROMPTS["quality-v1"], PROMPTS["quality-v2"]
    line = " Length and amount of detail are not merits in themselves."
    assert line in v2.system and line not in v1.system
    assert v2.system.replace(line, "") == v1.system and v2.user == v1.user


def test_length_counts_what_the_quality_judge_reads():
    assert field_words(IDEA) == {
        "problem_gap": 4,  # the label does not count
        "core_claim": 3,
        "method": 4,
        "inspiration": 7,
    }
    assert judged_words(IDEA) == 4 + 3 + 4


def test_ideation_rejects_k_below_one():
    with pytest.raises(ValueError):
        ideation_messages(PAPERS, 0)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("[Evidence Gap] Nobody.", "Nobody."),
        ("  [Off List Thing]: Body", "Body"),
        ("[Other] \u2014 Body", "Body"),
        ("No label here.", "No label here."),
        ("Body with [brackets] inside.", "Body with [brackets] inside."),
    ],
)
def test_strip_leading_label(text, expected):
    assert strip_leading_label(text) == expected


def test_quality_card_has_no_labels_and_no_inspiration():
    card = render_card(IDEA)
    assert card == (
        "Problem: Nobody has measured X.\nClaim: X predicts Y.\nApproach: Measure X across Z."
    )
    assert "[" not in card and "Paper 1's" not in card


def test_quality_orders_swap_the_ideas():
    first = quality_messages(IDEA, " Gold insight. ", idea_first=True)[1]["content"]
    second = quality_messages(IDEA, " Gold insight. ", idea_first=False)[1]["content"]
    assert first == f"Idea A:\n{render_card(IDEA)}\n\nIdea B:\nGold insight."
    assert second == f"Idea A:\nGold insight.\n\nIdea B:\n{render_card(IDEA)}"


def test_quality_against_an_anchor_shows_both_ideas_the_same_way():
    anchor = IdeaCard(
        problem_gap="[Other] Anchor gap.",
        core_claim="Anchor claim.",
        method="[Other] Anchor method.",
        inspiration="Never shown.",
    )
    user = quality_messages(IDEA, render_card(anchor), idea_first=False)[1]["content"]
    assert user == (
        "Idea A:\nProblem: Anchor gap.\nClaim: Anchor claim.\nApproach: Anchor method.\n\n"
        f"Idea B:\n{render_card(IDEA)}"
    )


def test_quality_prompt_shows_no_date():
    system = PROMPTS["quality-v1"].system.lower()
    assert "date" not in system and "year" not in system


@pytest.mark.parametrize(
    "text, winner",
    [('{"winner": "A"}', "A"), ('```json\n{"winner": "B"}\n```', "B")],
)
def test_parse_winner(text, winner):
    assert parse_winner(text) == winner


@pytest.mark.parametrize("text", ['{"winner": "a"}', '{"winner": "tie"}', "A", "[]", None])
def test_parse_winner_rejects_anything_else(text):
    with pytest.raises(JudgeOutputError):
        parse_winner(text)


@pytest.mark.parametrize(
    "idea_first, gold_first, q",
    [("A", "B", 1.0), ("A", "A", 0.5), ("B", "B", 0.5), ("B", "A", 0.0)],
)
def test_q_from_orders(idea_first, gold_first, q):
    assert q_from_orders(idea_first, gold_first) == q


def test_distance_keeps_fields_as_generated():
    user = distance_messages(IDEA, IDEA)[1]["content"]
    assert "problem_gap: [Evidence Gap] Nobody has measured X." in user
    assert "method: [Empirical Mapping]: Measure X across Z." in user
    assert "inspiration" not in user


def test_parse_similarity_and_distance():
    s = parse_similarity('{"problem_gap": 5, "core_claim": 1, "method": 3}')
    assert s == {"problem_gap": 5, "core_claim": 1, "method": 3}
    assert distance(s) == pytest.approx((0 + 1 + 0.5) / 3)
    assert distance({"problem_gap": 1, "core_claim": 1, "method": 1}) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        '{"problem_gap": 5, "core_claim": 1}',
        '{"problem_gap": 0, "core_claim": 1, "method": 3}',
        '{"problem_gap": 2.5, "core_claim": 1, "method": 3}',
        '{"problem_gap": true, "core_claim": 1, "method": 3}',
    ],
)
def test_parse_similarity_rejects_bad_scores(text):
    with pytest.raises(JudgeOutputError):
        parse_similarity(text)


def test_grounding_puts_papers_first_and_keeps_fields():
    user = grounding_messages(PAPERS, IDEA)[1]["content"]
    assert user.startswith("<papers>")
    assert user.endswith("Inspiration:\nPaper 1's X and Paper 2's Y.\n\nCore claim:\nX predicts Y.")


def test_parse_grounding():
    assert parse_grounding('{"score": 4}') == 4
    assert grounding(1) == 0.0 and grounding(5) == 1.0
    with pytest.raises(JudgeOutputError):
        parse_grounding('{"score": 6}')
