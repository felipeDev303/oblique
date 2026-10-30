"""grounding-v1: how much an idea's core claim depends on its sources.

The judge sees the two source papers, then the idea's inspiration and core
claim exactly as generated, and rates dependence from 1 to 5.
grounding = (s - 1) / 4. The papers come first so the K ideas of an instance
share the prefix.
"""

from __future__ import annotations

from oblique.generate.schema import IdeaCard
from oblique.prompts.base import Papers, Prompt, parse_json_object, parse_score, render_papers

SYSTEM = """You assess how much a research idea depends on its source papers. You will get summaries of two papers, then an idea's inspiration note and its core claim.

Rate from 1 to 5 how strongly the core claim depends on the inspiration and the source papers:
1 = no dependence: the claim could be made without these papers;
2 = weak: the papers set the topic, but the claim does not use their content;
3 = partial: the claim uses some content of the papers, but much of it comes from elsewhere;
4 = strong: the claim is mostly built from content of the papers;
5 = full: the claim follows directly from combining specific content of the papers, as the inspiration describes.

Judge the actual dependence on what the papers say, not whether the inspiration note asserts it.

Respond with one JSON object and nothing else: {"score": <1-5>}."""

USER = """{papers}

Inspiration:
{inspiration}

Core claim:
{core_claim}"""

GROUNDING_V1 = Prompt("grounding-v1", SYSTEM, USER)


def grounding_messages(
    papers: Papers, idea: IdeaCard, prompt: Prompt = GROUNDING_V1
) -> list[dict[str, str]]:
    return prompt.messages(
        papers=render_papers(papers), inspiration=idea.inspiration, core_claim=idea.core_claim
    )


def parse_grounding(text: str | None) -> int:
    return parse_score(parse_json_object(text), "score")


def grounding(score: int) -> float:
    return (score - 1) / 4
