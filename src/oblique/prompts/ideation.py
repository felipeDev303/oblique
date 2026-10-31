"""ideation-v3: K distinct research ideas from two source papers, JSON only.

Each idea has the four fields of `IdeaCard`. `problem_gap` opens with an
opportunity-pattern label and `method` with a method-paradigm label, from the
canonical lists in `oblique.generate.labels`, each defined in one line written
for this project. The request is always "Propose {k} distinct research
idea(s)", the same phrase in every condition. Output: {"ideas": [...]}.
The papers come first so every call on an instance shares the same prefix.

v3 (ADR-009) caps the words of each field and adds the fixed text that conditions
a sequential idea on the earlier ones; the first idea of a sequence reads exactly
like any other K = 1 call. Exceeding a cap does not make an idea invalid (ADR-006).

v2 asked for one to three sentences per field and stays for the runs before
ADR-009. v1 (never sent to the API) asked for "{k} research ideas" and listed the
labels without definitions.
"""

from __future__ import annotations

from collections.abc import Sequence

from oblique.generate.schema import IdeaCard
from oblique.prompts.base import Papers, Prompt, render_papers

OPPORTUNITY_DEFINITIONS = {
    "Puzzle / Contradiction": "results or assumptions that conflict with each other or cannot both hold.",
    "Explanation Gap": "an observed effect or result whose cause or mechanism is still unexplained.",
    "Scope Mismatch": "something established in one setting that is untested or ill-suited in another.",
    "Evidence Gap": "a claim that is plausible or widely assumed but lacks direct support.",
    "Bridge Opportunity": "ideas from the two papers that have not been connected but could inform each other.",
    "Failure / Risk Gap": "a known failure mode, weakness or risk that current approaches leave open.",
    "Resource Bottleneck": "progress held back by the cost of data, compute, time, labor or equipment.",
    "Other": "none of the above fits.",
}

METHOD_DEFINITIONS = {
    "Synthesis / Unification": "combine methods or theories into one framework that covers both.",
    "Relax / Extend Scope": "drop an assumption, or carry an approach to new settings, scales or populations.",
    "Robustification": "make a method or result hold under noise, shifts, adversaries or weaker assumptions.",
    "Formal Derivation": "establish the claim by proof, derivation or a formal model.",
    "Empirical Mapping": "measure, survey or experiment systematically to characterize a phenomenon.",
    "Artifact / System": "build a tool, dataset, benchmark, system or device.",
    "Optimization / Search": "improve an objective by searching over or tuning designs, parameters or policies.",
    "Other": "none of the above fits.",
}


def _label_list(definitions: dict[str, str]) -> str:
    return "\n".join(f"- [{label}]: {text}" for label, text in definitions.items())


SYSTEM = f"""You are a research scientist. You will read summaries of two research papers and propose new research ideas that build on them.

Each idea is a JSON object with exactly these four string fields:
- "problem_gap": the open problem or gap the idea addresses. Start with one opportunity-pattern label in square brackets, then describe the gap.
- "core_claim": the central hypothesis or contribution, stated specifically enough that it could turn out to be wrong.
- "method": how the claim would be developed or tested. Start with one method-paradigm label in square brackets, then describe the approach.
- "inspiration": which specific elements of which paper led to the idea, and how they are combined.

Opportunity patterns (for "problem_gap"):
{_label_list(OPPORTUNITY_DEFINITIONS)}

Method paradigms (for "method"):
{_label_list(METHOD_DEFINITIONS)}

Copy each label exactly as listed, brackets included, without its definition. Use [Other] only when no listed label fits.
Keep each field to one to three sentences.

Respond with one JSON object and nothing else: {{"ideas": [ ... ]}}. No code fences, no text before or after the JSON."""

USER = """{papers}

Propose {k} distinct research {noun} based on these papers. The "ideas" array must contain exactly {k} {objects}."""

IDEATION_V2 = Prompt("ideation-v2", SYSTEM, USER)

WORD_LIMITS = {"problem_gap": 40, "core_claim": 40, "method": 60, "inspiration": 80}

_V2_LENGTH_LINE = "Keep each field to one to three sentences."
_V3_LENGTH_LINE = (
    "Word limits per field, not counting the label: "
    + ", ".join(f'"{field}" {limit}' for field, limit in WORD_LIMITS.items())
    + ". Stay within them."
)

USER_V3 = """{papers}

{previous}Propose {k} distinct research {noun} based on these papers. The "ideas" array must contain exactly {k} {objects}."""

# The fixed text of sequential generation (ADR-009), the same in every condition.
PREVIOUS_BLOCK = """Ideas already proposed from these papers:

{ideas}

The new idea must be distinct from these.

"""

PREVIOUS_IDEA = """Idea {number}:
problem_gap: {problem_gap}
core_claim: {core_claim}
method: {method}
inspiration: {inspiration}"""

IDEATION_V3 = Prompt(
    "ideation-v3",
    SYSTEM.replace(_V2_LENGTH_LINE, _V3_LENGTH_LINE),
    USER_V3,
    extra=(("previous_block", PREVIOUS_BLOCK), ("previous_idea", PREVIOUS_IDEA)),
)


# The frame around a card (Arquitectura): after the documents, before the task. A card is one
# more fixed block, so it has its own version and ideation-v3 itself does not change.
INJECT_TEXT = """Before writing this idea, apply the following oblique constraint to how you approach the documents: «{card}». Do not mention the constraint in the idea.

"""
INJECT_V1 = Prompt("inject-v1", "", INJECT_TEXT)


def render_previous(previous: Sequence[IdeaCard]) -> str:
    """The earlier ideas of a sequential set, exactly as generated; empty for the first idea."""
    if not previous:
        return ""
    ideas = "\n\n".join(
        PREVIOUS_IDEA.format(number=i, **idea.model_dump()) for i, idea in enumerate(previous, 1)
    )
    return PREVIOUS_BLOCK.format(ideas=ideas)


def ideation_messages(
    papers: Papers,
    k: int,
    prompt: Prompt = IDEATION_V3,
    previous: Sequence[IdeaCard] = (),
    card: str | None = None,
) -> list[dict[str, str]]:
    """K ideas in one call; with `previous`, one more idea of a sequential set; with `card`,
    one idea written under that card."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if (previous or card is not None) and (k != 1 or "{previous}" not in prompt.user):
        raise ValueError(f"a sequence or a card needs k = 1 and a {{previous}} slot in {prompt.id}")
    constraint = "" if card is None else INJECT_V1.user.format(card=card)
    return prompt.messages(
        papers=render_papers(papers),
        previous=render_previous(previous) + constraint,
        k=str(k),
        noun="idea" if k == 1 else "ideas",
        objects="object" if k == 1 else "objects",
    )
