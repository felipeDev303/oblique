"""evolve-v1: Muse Wonder writes new deck cards from the measured map (ADR-010).

The objective and the deck rules come from ADR-010. The map goes in as data: every seed and
paper card with its measures, and the random-noun controls only as an aggregate, so their words
cannot become material for new cards. No Oblique Strategies text is in this prompt (ADR-002):
the deck is only named, to say that its cards must not be copied or paraphrased.
"""

from __future__ import annotations

from oblique.prompts.base import Prompt

SYSTEM = """You write cards for a deck used in research ideation. A card is a short constraint on how to approach source documents. A model reads summaries of two research papers and proposes one research idea under one card. Cards are applied to pairs of papers from 16 research domains.

The model receives a card in this frame, after the papers and before the task:
"Before writing this idea, apply the following oblique constraint to how you approach the documents: «card». Do not mention the constraint in the idea."

Every card below was measured on the same 50 pairs of papers, one idea per pair, against a re-sample with no card (the reference):
- dq: how often the idea beats a reference idea from the same model in a pairwise quality judgment, at equal length, minus the same value for the no-card re-sample.
- dd: distance from the idea to a set of three no-card ideas on the same papers (0 to 1), minus the same value for the no-card re-sample.
- grounding: how much the idea depends on the source papers (0 to 1).
Brackets are 95% confidence intervals.

Bands, in this order of priority:
- destructive: the upper bound of dq is below -0.05.
- liminal: the lower bound of dd is above 0.
- inert: the interval of dd contains 0.
- convergent: the upper bound of dd is below 0.

Rules for each new card:
- 5 to 15 words, in English.
- It names no technique: no specific method, tool, model or statistical procedure.
- It belongs to exactly one of these families: subtraction, inversion, scale, role, constraint, ambiguity_specificity, error, time.
- It is neither a copy nor a paraphrase of any card in the table, nor of any card of Oblique Strategies by Brian Eno and Peter Schmidt.

Respond with one JSON object and nothing else: {"cards": [ ... ]}. No code fences, no text before or after the JSON. Each element has exactly these fields:
- "text": the card.
- "family": its family.
- "parents": the IDs of the cards in the table it takes inspiration from, at least one.
- "justification": one line on why it should raise distance without lowering quality or grounding."""

USER = """Reference, no card: {reference}

Measured cards (E6 cards have no family):

{table}

Controls, not listed: {controls}

Propose {n} new cards that aim to raise distance (dd) without lowering quality (dq) or grounding. The "cards" array must contain exactly {n} objects."""

EVOLVE_V1 = Prompt("evolve-v1", SYSTEM, USER)
