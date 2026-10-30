import math

import numpy as np
import pytest

from oblique.metrics import (
    bootstrap_mean_ci,
    creative_utility,
    greedy_order,
    paired_delta_ci,
)

NAN = float("nan")


def _sym(k: int, pairs: dict[tuple[int, int], float]) -> list[list[float]]:
    D = [[0.0] * k for _ in range(k)]
    for (i, j), d in pairs.items():
        D[i][j] = D[j][i] = d
    return D


# --- Creative Utility -------------------------------------------------------


def test_hand_computed_example():
    # q = (1, 0.5, 1); d01 = 0.4, d02 = 0.8, d12 = 0.6
    # greedy: idea 0 (q=1, marginal 1)
    #         idea 2 (1 * 0.8 = 0.8 beats 0.5 * 0.4 = 0.2)
    #         idea 1 (0.5 * min(0.4, 0.6) = 0.2)
    # CU = (1 + 0.9 * 0.8 + 0.81 * 0.2) / (1 + 0.9 + 0.81) = 1.882 / 2.71
    q = [1.0, 0.5, 1.0]
    D = _sym(3, {(0, 1): 0.4, (0, 2): 0.8, (1, 2): 0.6})
    order, marginals = greedy_order(q, D)
    assert order == [0, 2, 1]
    assert marginals == pytest.approx([1.0, 0.8, 0.2])
    assert creative_utility(q, D) == pytest.approx(1.882 / 2.71)


def test_paper_figure_1_reward():
    # Wadhwa et al. fig. 1: q = (1, 0, 0.5), min-distances (1, 0.4, 0.2) -> R = 1.10
    # without discount or normalisation. d(u3, u2) only needs to be >= 0.2.
    q = [1.0, 0.0, 0.5]
    D = _sym(3, {(0, 1): 0.4, (0, 2): 0.2, (1, 2): 0.5})
    assert creative_utility(q, D, gamma=1.0) * 3 == pytest.approx(1.10)


def test_independent_of_generation_order():
    q = [1.0, 0.5, 1.0]
    D = _sym(3, {(0, 1): 0.4, (0, 2): 0.8, (1, 2): 0.6})
    perm = [2, 0, 1]
    q_p = [q[i] for i in perm]
    D_p = [[D[i][j] for j in perm] for i in perm]
    assert creative_utility(q_p, D_p) == pytest.approx(creative_utility(q, D))


def test_bounds():
    assert creative_utility([1, 1, 1], _sym(3, {(0, 1): 1, (0, 2): 1, (1, 2): 1})) == 1.0
    assert creative_utility([0, 0, 0], _sym(3, {(0, 1): 1, (0, 2): 1, (1, 2): 1})) == 0.0


def test_redundant_ideas_add_nothing():
    # three copies of one good idea: only the first counts
    D = _sym(3, {})
    assert creative_utility([1, 1, 1], D) == pytest.approx(1 / 2.71)


def test_single_idea_is_its_quality():
    assert creative_utility([0.5], [[0.0]]) == 0.5


def test_zero_quality_ideas_may_carry_nan_distances():
    # idea 1 is invalid (q = 0): its distances are unknown
    q = [1.0, 0.0, 0.5]
    D = [[0.0, NAN, 0.6], [NAN, 0.0, NAN], [0.6, NAN, 0.0]]
    assert creative_utility(q, D) == pytest.approx((1 + 0.9 * 0.3) / 2.71)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        creative_utility([], [])
    with pytest.raises(ValueError):
        creative_utility([1, 1], [[0.0]])
    with pytest.raises(ValueError):
        creative_utility([1.5], [[0.0]])
    with pytest.raises(ValueError):
        creative_utility([1, 1], [[0, NAN], [NAN, 0]])


# --- Bootstrap --------------------------------------------------------------


def test_constant_values_give_degenerate_interval():
    ci = bootstrap_mean_ci([0.5] * 50)
    assert (ci.mean, ci.lo, ci.hi, ci.n) == (0.5, 0.5, 0.5, 50)


def test_percentile_interval_matches_normal_approximation():
    # 50 values of 0 and 1: mean 0.5, se = 0.5 / sqrt(50) ~ 0.0707 -> CI ~ 0.5 +- 0.139
    ci = bootstrap_mean_ci([0.0, 1.0] * 25)
    assert ci.mean == 0.5
    assert ci.lo == pytest.approx(0.5 - 1.96 * 0.5 / math.sqrt(50), abs=0.02)
    assert ci.hi == pytest.approx(0.5 + 1.96 * 0.5 / math.sqrt(50), abs=0.02)


def test_default_seed_is_reproducible_and_seed_matters():
    x = np.random.default_rng(1).random(50)
    assert bootstrap_mean_ci(x) == bootstrap_mean_ci(x)
    other = bootstrap_mean_ci(x, seed=123)
    assert other.mean == bootstrap_mean_ci(x).mean
    assert other.lo != bootstrap_mean_ci(x).lo
    # stability check: 10,000 resamples agree to ~2 decimals across seeds
    assert other.lo == pytest.approx(bootstrap_mean_ci(x).lo, abs=0.01)


def test_pairing_removes_between_instance_variance():
    # a constant +0.1 effect on wildly different instances: paired CI is exactly 0.1
    base = np.random.default_rng(0).random(50)
    a = {f"i-{n:04d}": b + 0.1 for n, b in enumerate(base)}
    c = {f"i-{n:04d}": b for n, b in enumerate(base)}
    ci = paired_delta_ci(a, c)
    assert ci.mean == pytest.approx(0.1)
    assert ci.lo == pytest.approx(0.1)
    assert ci.hi == pytest.approx(0.1)
    assert ci.excludes(0.0)


def test_paired_delta_is_independent_of_dict_order():
    a = {"i-0001": 0.9, "i-0002": 0.1, "i-0003": 0.5}
    b = {"i-0003": 0.2, "i-0001": 0.4, "i-0002": 0.3}
    assert paired_delta_ci(a, b) == paired_delta_ci(dict(reversed(a.items())), b)


def test_unpaired_instances_are_rejected():
    with pytest.raises(ValueError, match="unpaired"):
        paired_delta_ci({"i-0001": 1.0}, {"i-0002": 1.0})


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.1, NAN])
