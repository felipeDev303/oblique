"""ideation-v1: K research ideas from two source papers, JSON only.

Each idea has the four fields of `IdeaCard`. `problem_gap` opens with an
opportunity-pattern label and `method` with a method-paradigm label, from the
canonical lists in `oblique.generate.labels`. Output: {"ideas": [...]}.
The papers come first so every call on an instance shares the same prefix.
"""

from __future__ import annotations

from oblique.generate.labels import METHOD_PARADIGMS, OPPORTUNITY_PATTERNS
from oblique.prompts.base import Papers, Prompt, render_papers


def _label_list(labels: tuple[str, ...]) -> str:
    return "\n".join(f"- [{label}]" for label in labels)


SYSTEM = f"""You are a research scientist. You will read summaries of two research papers and propose new research ideas that build on them.

Each idea is a JSON object with exactly these four string fields:
- "problem_gap": the open problem or gap the idea addresses. Start with one opportunity-pattern label in square brackets, then describe the gap.
- "core_claim": the central hypothesis or contribution, stated specifically enough that it could turn out to be wrong.
- "method": how the claim would be developed or tested. Start with one method-paradigm label in square brackets, then describe the approach.
- "inspiration": which specific elements of which paper led to the idea, and how they are combined.

Opportunity patterns (for "problem_gap"):
{_label_list(OPPORTUNITY_PATTERNS)}

Method paradigms (for "method"):
{_label_list(METHOD_PARADIGMS)}

Copy each label exactly as listed, brackets included. Use [Other] only when no listed label fits.
Keep each field to one to three sentences.

Respond with one JSON object and nothing else: {{"ideas": [ ... ]}}. No code fences, no text before or after the JSON."""

USER = """{papers}

Propose {count} based on these papers. The "ideas" array must contain exactly {k} {objects}."""

IDEATION_V1 = Prompt("ideation-v1", SYSTEM, USER)


def ideation_messages(papers: Papers, k: int, prompt: Prompt = IDEATION_V1) -> list[dict[str, str]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    return prompt.messages(
        papers=render_papers(papers),
        count="one research idea" if k == 1 else f"{k} research ideas",
        k=str(k),
        objects="object" if k == 1 else "objects",
    )
