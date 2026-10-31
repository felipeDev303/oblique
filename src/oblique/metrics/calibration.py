"""Judge calibration without an external reference (ADR-007): position consistency,
length bias and agreement between judges. Pure functions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

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


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_condition_logit(
    condition: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_conditions: int,
    *,
    ridge: float = 1e-6,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """logit P(y) = alpha[condition] + beta * x, by Newton steps. Returns (alpha, beta).

    With one intercept per condition the Hessian is diagonal plus one row and column, so each
    step costs O(rows), which is what lets 48 conditions be bootstrapped 10,000 times. The small
    ridge and the bounded step keep the fit finite when a resample separates the classes (the
    rates then saturate at 0 or 1).
    """
    alpha, beta = np.zeros(n_conditions), 0.0
    for _ in range(max_iter):
        p = sigmoid(alpha[condition] + beta * x)
        w, r = p * (1 - p), y - p
        g_alpha = np.bincount(condition, r, n_conditions) - ridge * alpha
        g_beta = float(x @ r) - ridge * beta
        h_alpha = np.bincount(condition, w, n_conditions) + ridge
        h_cross = np.bincount(condition, w * x, n_conditions)
        schur = float((w * x) @ x) + ridge - float(h_cross @ (h_cross / h_alpha))
        d_beta = (g_beta - float(h_cross @ (g_alpha / h_alpha))) / schur
        d_alpha = (g_alpha - h_cross * d_beta) / h_alpha
        d_alpha, d_beta = np.clip(d_alpha, -5, 5), float(np.clip(d_beta, -5, 5))
        alpha, beta = alpha + d_alpha, beta + d_beta
        if max(np.abs(d_alpha).max(), abs(d_beta)) < tol:
            break
    return alpha, beta


def controlled_win_rates(
    rows: Sequence[dict[str, Any]],
    conditions: Sequence[str],
    *,
    comparisons: Sequence[tuple[str, str]] = (),
    words_per_unit: float = 10.0,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | np.random.SeedSequence | None = None,
) -> dict[str, Any]:
    """Win rate controlled for length against the anchor (ADR-009, point 7).

    logit P(win) = a_condition + b * (words(idea) - words(anchor)) / words_per_unit, fitted jointly
    on one binary row per judgment (`instance`, `condition`, `win`, `delta_words`). The
    controlled rate of a condition is the fitted P(win) at equal length (delta = 0).
    Percentile CIs resample instances, since the ideas of an instance share one anchor.
    """
    from oblique.seeds import seed_sequence

    conditions = list(conditions)
    index = {c: i for i, c in enumerate(conditions)}
    instances = sorted({r["instance"] for r in rows})
    condition = np.array([index[r["condition"]] for r in rows], dtype=int)
    x = np.array([r["delta_words"] / words_per_unit for r in rows], dtype=float)
    y = np.array([float(r["win"]) for r in rows])
    members: dict[str, list[int]] = {i: [] for i in instances}
    for j, r in enumerate(rows):
        members[r["instance"]].append(j)
    pairs = [(index[a], index[b]) for a, b in comparisons]

    def summarize(idx: np.ndarray | slice) -> np.ndarray:
        alpha, beta = fit_condition_logit(condition[idx], x[idx], y[idx], len(conditions))
        rates = sigmoid(alpha)
        return np.array([*rates, *(rates[a] - rates[b] for a, b in pairs), beta], dtype=float)

    point = summarize(slice(None))
    gen = np.random.default_rng(seed_sequence("bootstrap") if seed is None else seed)
    draws = np.empty((n_boot, len(point)))
    groups = [np.array(members[i], dtype=int) for i in instances]
    for b in range(n_boot):
        idx = np.concatenate([groups[g] for g in gen.integers(0, len(groups), len(groups))])
        draws[b] = summarize(idx)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    cell = [
        {"mean": float(p), "lo": float(a), "hi": float(b), "n": len(instances)}
        for p, a, b in zip(point, lo, hi, strict=True)
    ]
    k = len(conditions)
    return {
        "rates": dict(zip(conditions, cell[:k], strict=True)),
        "differences": {f"{a}-{b}": c for (a, b), c in zip(comparisons, cell[k:-1], strict=True)},
        "log_odds_per_unit": cell[-1],
        "words_per_unit": words_per_unit,
        "judgments": len(rows),
    }


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
