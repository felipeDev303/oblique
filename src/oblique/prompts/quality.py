"""quality-v2: pairwise comparison of a generated idea against a reference idea.

v2 adds that length and detail are not merits in themselves (ADR-009); v1 stays for
the runs before it.

The judge sees two ideas, A and B, and names the stronger one. Each idea is
judged in both orders (idea as A, then as B), so q is in {0, 0.5, 1}. The
generated idea is shown as Problem / Claim / Approach (problem_gap, core_claim,
method) with its leading labels removed and without inspiration, because the
gold idea has neither (ADR-006 amendment 5). No date is shown.

The reference is the anchor idea, rendered the same way (primary q, ADR-008), or
the gold insight as it stands (secondary q).
"""

from __future__ import annotations

from typing import Literal

from oblique.generate.labels import strip_leading_label
from oblique.generate.schema import IdeaCard
from oblique.prompts.base import JudgeOutputError, Prompt, parse_json_object

Winner = Literal["A", "B"]

SYSTEM = """You are an expert reviewer. You will compare two research ideas that build on the same pair of source papers and decide which one is the stronger research contribution.

Weigh, together:
- importance: how much the problem matters to its field;
- novelty: how much the claim goes beyond what is already known or obvious;
- soundness: whether the approach could actually establish the claim;
- specificity: whether the idea is concrete enough to act on.

The ideas appear in random order; the order carries no information. One may be written as a single paragraph and the other in labeled parts: judge the content, not the format, the length or the style.

Respond with one JSON object and nothing else: {"winner": "A"} or {"winner": "B"}."""

USER = """Idea A:
{idea_a}

Idea B:
{idea_b}"""

QUALITY_V1 = Prompt("quality-v1", SYSTEM, USER)

_V1_STYLE_LINE = "judge the content, not the format, the length or the style."
QUALITY_V2 = Prompt(
    "quality-v2",
    SYSTEM.replace(
        _V1_STYLE_LINE,
        f"{_V1_STYLE_LINE} Length and amount of detail are not merits in themselves.",
    ),
    USER,
)


def render_card(idea: IdeaCard) -> str:
    return (
        f"Problem: {strip_leading_label(idea.problem_gap)}\n"
        f"Claim: {idea.core_claim.strip()}\n"
        f"Approach: {strip_leading_label(idea.method)}"
    )


def field_words(idea: IdeaCard) -> dict[str, int]:
    """Words per field, not counting the leading label."""
    return {
        "problem_gap": len(strip_leading_label(idea.problem_gap).split()),
        "core_claim": len(idea.core_claim.split()),
        "method": len(strip_leading_label(idea.method).split()),
        "inspiration": len(idea.inspiration.split()),
    }


def judged_words(idea: IdeaCard) -> int:
    """Length of an idea (ADR-009): words of gap + claim + method, what the quality judge reads."""
    words = field_words(idea)
    return words["problem_gap"] + words["core_claim"] + words["method"]


def quality_messages(
    idea: IdeaCard, reference: str, *, idea_first: bool, prompt: Prompt = QUALITY_V2
) -> list[dict[str, str]]:
    card, reference = render_card(idea), reference.strip()
    a, b = (card, reference) if idea_first else (reference, card)
    return prompt.messages(idea_a=a, idea_b=b)


def parse_winner(text: str | None) -> Winner:
    winner = parse_json_object(text).get("winner")
    if winner not in ("A", "B"):
        raise JudgeOutputError(f"'winner' must be 'A' or 'B', got {winner!r}")
    return winner


def idea_wins(winner: Winner, *, idea_first: bool) -> bool:
    return winner == ("A" if idea_first else "B")


def q_from_orders(idea_first: Winner, gold_first: Winner) -> float:
    """q in {0, 0.5, 1}: share of the two orders in which the generated idea wins."""
    wins = idea_wins(idea_first, idea_first=True) + idea_wins(gold_first, idea_first=False)
    return wins / 2
