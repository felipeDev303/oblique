"""distance-v1: field-by-field similarity of two generated ideas.

Similarity 1-5 on problem_gap, core_claim and method, with the fields exactly
as generated (labels included). d = mean over fields of (5 - s) / 4.
"""

from __future__ import annotations

from oblique.generate.schema import IdeaCard
from oblique.prompts.base import Prompt, parse_json_object, parse_score

FIELDS = ("problem_gap", "core_claim", "method")

SYSTEM = """You compare two research ideas field by field: problem_gap, core_claim and method.

For each field, rate how similar the two ideas are, from 1 to 5:
1 = unrelated;
2 = same broad area, different substance;
3 = related, with substantial differences;
4 = mostly the same, with minor differences;
5 = essentially the same.

Rate the substance, not the wording: two fields that say the same thing in different words are similar.

Respond with one JSON object and nothing else: {"problem_gap": <1-5>, "core_claim": <1-5>, "method": <1-5>}."""

USER = """Idea 1:
problem_gap: {gap_1}
core_claim: {claim_1}
method: {method_1}

Idea 2:
problem_gap: {gap_2}
core_claim: {claim_2}
method: {method_2}"""

DISTANCE_V1 = Prompt("distance-v1", SYSTEM, USER)


def distance_messages(
    first: IdeaCard, second: IdeaCard, prompt: Prompt = DISTANCE_V1
) -> list[dict[str, str]]:
    return prompt.messages(
        gap_1=first.problem_gap,
        claim_1=first.core_claim,
        method_1=first.method,
        gap_2=second.problem_gap,
        claim_2=second.core_claim,
        method_2=second.method,
    )


def parse_similarity(text: str | None) -> dict[str, int]:
    data = parse_json_object(text)
    return {f: parse_score(data, f) for f in FIELDS}


def distance(similarity: dict[str, int]) -> float:
    return sum((5 - similarity[f]) / 4 for f in FIELDS) / len(FIELDS)
