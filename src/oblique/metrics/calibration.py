"""Judge calibration without an external reference (ADR-007): position consistency,
length bias and agreement between judges. Pure functions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np


def ranks(x: Sequence[float]) -> np.ndarray:
    """Ranks starting at 1, ties get their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        r[tied] = r[tied].mean()
    return r


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("x and y differ in length")
    return pearson(ranks(x), ranks(y))


def position_consistency(verdicts: Sequence[tuple[str, str]]) -> dict[str, float | int]:
    """verdicts[i] = (winner with the idea shown as A, winner with the idea shown as B).

    Consistent: the same idea wins in both orders, i.e. (A, B) or (B, A).
    "A both" / "B both": the same position wins in both orders (position bias).
    """
    n = len(verdicts)
    if n == 0:
        raise ValueError("no verdicts")
    count = Counter(verdicts)
    consistent = count[("A", "B")] + count[("B", "A")]
    return {
        "n": n,
        "consistent": consistent / n,
        "a_both": count[("A", "A")] / n,
        "b_both": count[("B", "B")] / n,
    }


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float | None:
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def agreement(a: Sequence[object], b: Sequence[object]) -> dict[str, float | int | None]:
    """Exact agreement and Cohen's kappa between two judges' labels on the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    exact = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    return {"n": len(a), "exact": exact, "kappa": cohen_kappa(a, b)}
