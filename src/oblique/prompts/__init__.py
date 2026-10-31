from oblique.prompts.base import JudgeOutputError, Prompt, render_papers
from oblique.prompts.distance import DISTANCE_V1, distance, distance_messages, parse_similarity
from oblique.prompts.grounding import GROUNDING_V1, grounding, grounding_messages, parse_grounding
from oblique.prompts.ideation import IDEATION_V2, IDEATION_V3, WORD_LIMITS, ideation_messages
from oblique.prompts.quality import (
    QUALITY_V1,
    QUALITY_V2,
    field_words,
    judged_words,
    parse_winner,
    q_from_orders,
    quality_messages,
    render_card,
)

PROMPTS = {
    p.id: p for p in (IDEATION_V2, IDEATION_V3, QUALITY_V1, QUALITY_V2, DISTANCE_V1, GROUNDING_V1)
}

__all__ = [
    "DISTANCE_V1",
    "GROUNDING_V1",
    "IDEATION_V2",
    "IDEATION_V3",
    "PROMPTS",
    "QUALITY_V1",
    "QUALITY_V2",
    "WORD_LIMITS",
    "JudgeOutputError",
    "Prompt",
    "distance",
    "distance_messages",
    "field_words",
    "grounding",
    "grounding_messages",
    "ideation_messages",
    "judged_words",
    "parse_grounding",
    "parse_similarity",
    "parse_winner",
    "q_from_orders",
    "quality_messages",
    "render_card",
    "render_papers",
]
