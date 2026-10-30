"""Bootstrap confidence intervals over benchmark instances.

Percentile intervals of the mean, 10,000 resamples, not stratified.
Paired comparisons first take the per-instance difference and then resample
those differences, so every resample uses the same instances in both conditions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from oblique.seeds import seed_sequence

N_BOOT = 10_000
ALPHA = 0.05


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float
    n: int

    def excludes(self, value: float) -> bool:
        return self.lo > value or self.hi < value


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int | np.random.SeedSequence | None = None,
) -> CI:
    """Percentile CI of the mean of one value per instance.

    `seed` defaults to the pre-registered bootstrap stream (see `oblique.seeds`).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("need a non-empty 1-D array with one value per instance")
    if np.isnan(x).any():
        raise ValueError("NaN in bootstrap input")
    rng = np.random.default_rng(seed_sequence("bootstrap") if seed is None else seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=len(x))


def paired_differences(a: Mapping[str, float], b: Mapping[str, float]) -> np.ndarray:
    """a[i] - b[i] per instance, in a fixed (sorted) instance order."""
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(f"unpaired instances: only in a {only_a[:5]}, only in b {only_b[:5]}")
    return np.array([a[i] - b[i] for i in sorted(a)], dtype=float)


def paired_delta_ci(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int | np.random.SeedSequence | None = None,
) -> CI:
    """CI of mean(a - b), paired by instance id (e.g. CU(ML) - CU(C1))."""
    return bootstrap_mean_ci(paired_differences(a, b), n_boot=n_boot, alpha=alpha, seed=seed)
