"""Secondary distance from local sentence embeddings (author, 2026-10-30).

Model: sentence-transformers/all-mpnet-base-v2 at a pinned revision (Apache 2.0).
Per field (problem_gap, core_claim, method, exactly as generated):
d_field = clip(1 - cos, 0, 1); the pair distance is the mean over the three
fields. The LLM judge's distance is the primary d; this one is used for the
Spearman check and the visualization.

Needs the `embed` dependency group: `uv run --group embed ...`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oblique.generate.schema import IdeaCard

MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
FIELDS = ("problem_gap", "core_claim", "method")


def field_distance(u: np.ndarray, v: np.ndarray) -> float:
    """clip(1 - cos(u, v), 0, 1)."""
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.clip(1.0 - cos, 0.0, 1.0))


def pair_distance(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    return sum(field_distance(a[f], b[f]) for f in FIELDS) / len(FIELDS)


class Embedder:
    def __init__(self, model: str = MODEL, revision: str = REVISION, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, revision=revision, device=device)

    def embed(self, ideas: Sequence[IdeaCard]) -> list[dict[str, np.ndarray]]:
        """One {field: vector} per idea."""
        texts = [getattr(idea, f) for idea in ideas for f in FIELDS]
        vectors = self.model.encode(texts, batch_size=32, convert_to_numpy=True)
        n = len(FIELDS)
        return [
            dict(zip(FIELDS, vectors[i * n : (i + 1) * n], strict=True)) for i in range(len(ideas))
        ]
