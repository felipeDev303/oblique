from oblique.prompts.base import JudgeOutputError, Prompt, render_papers
from oblique.prompts.distance import DISTANCE_V1, distance, distance_messages, parse_similarity
from oblique.prompts.grounding import GROUNDING_V1, grounding, grounding_messages, parse_grounding
from oblique.prompts.ideation import IDEATION_V1, ideation_messages
from oblique.prompts.quality import (
    QUALITY_V1,
    parse_winner,
    q_from_orders,
    quality_messages,
    render_card,
)

PROMPTS = {p.id: p for p in (IDEATION_V1, QUALITY_V1, DISTANCE_V1, GROUNDING_V1)}

__all__ = [
    "DISTANCE_V1",
    "GROUNDING_V1",
    "IDEATION_V1",
    "PROMPTS",
    "QUALITY_V1",
    "JudgeOutputError",
    "Prompt",
    "distance",
    "distance_messages",
    "grounding",
    "grounding_messages",
    "ideation_messages",
    "parse_grounding",
    "parse_similarity",
    "parse_winner",
    "q_from_orders",
    "quality_messages",
    "render_card",
    "render_papers",
]
