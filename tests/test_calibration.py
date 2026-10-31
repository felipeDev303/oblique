import numpy as np
import pytest

from oblique.judges.embed import field_distance, pair_distance
from oblique.metrics.calibration import (
    agreement,
    cohen_kappa,
    controlled_win_rates,
    logistic_fit,
    pearson,
    position_consistency,
    ranks,
    spearman,
)


def test_ranks_average_ties():
    assert ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]


def test_spearman_is_rank_correlation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 1000]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        spearman([1, 2], [1])


def test_pearson_needs_variance_and_three_points():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2], [1, 2]) is None


def test_position_consistency():
    verdicts = [("A", "B"), ("B", "A"), ("A", "A"), ("B", "B")]
    assert position_consistency(verdicts) == {
        "n": 4,
        "consistent": 0.5,
        "a_both": 0.25,
        "b_both": 0.25,
    }


def test_agreement_and_kappa():
    a = [1.0, 0.5, 0.0, 1.0]
    assert agreement(a, a) == {"n": 4, "exact": 1.0, "kappa": 1.0}
    # Hand-computed: observed 0.5, expected (1*2 + 1*1 + 2*1) / 16 = 5/16.
    b = [1.0, 0.0, 0.0, 0.5]
    assert cohen_kappa(a, b) == pytest.approx((0.5 - 5 / 16) / (1 - 5 / 16))
    assert cohen_kappa([1, 1], [1, 1]) is None


def test_embedding_distance_is_clipped_one_minus_cosine():
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert field_distance(u, u) == pytest.approx(0.0)
    assert field_distance(u, v) == pytest.approx(1.0)
    assert field_distance(u, -u) == 1.0  # cos = -1 -> clipped
    a = {"problem_gap": u, "core_claim": u, "method": u}
    b = {"problem_gap": u, "core_claim": v, "method": -u}
    assert pair_distance(a, b) == pytest.approx(2 / 3)


def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20_000)
    X = np.column_stack([np.ones_like(x), x])
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-(0.5 + 1.2 * x)))).astype(float)
    assert logistic_fit(X, y) == pytest.approx([0.5, 1.2], abs=0.06)


def test_logistic_fit_stays_finite_under_separation():
    X = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    beta = logistic_fit(X, np.array([0.0, 0, 0, 1, 1, 1]))
    assert np.isfinite(beta).all() and beta[1] > 3


def _rows(rate_by_condition, delta=lambda j: 0):
    """Ten judgments per instance and condition, wins at the given rate."""
    return [
        {
            "instance": f"i-{i}",
            "condition": cond,
            "win": j < round(rate * 10),
            "delta_words": delta(j),
        }
        for i in range(20)
        for cond, rate in rate_by_condition.items()
        for j in range(10)
    ]


def test_controlled_rate_equals_the_raw_rate_when_length_is_equal():
    out = controlled_win_rates(
        _rows({"C0": 0.5, "C1s": 0.3}),
        ["C0", "C1s"],
        comparisons=[("C1s", "C0")],
        n_boot=50,
        seed=1,
    )
    assert out["rates"]["C0"]["mean"] == pytest.approx(0.5)
    assert out["rates"]["C1s"]["mean"] == pytest.approx(0.3)
    assert out["differences"]["C1s-C0"]["mean"] == pytest.approx(-0.2)
    assert out["judgments"] == 400 and out["rates"]["C0"]["n"] == 20


def test_controlling_removes_a_gap_that_is_only_length():
    # The same probability curve in length for both conditions; C1 is just shorter.
    rng = np.random.default_rng(3)
    data = []
    for i in range(40):
        for cond, center in (("C0", 0), ("C1", -30)):
            for d in rng.normal(center, 15, size=50):
                win = rng.random() < 1 / (1 + np.exp(-d / 10))
                data.append({"instance": f"i-{i}", "condition": cond, "win": win, "delta_words": d})
    raw = {c: np.mean([r["win"] for r in data if r["condition"] == c]) for c in ("C0", "C1")}
    out = controlled_win_rates(data, ["C0", "C1"], comparisons=[("C1", "C0")], n_boot=20, seed=1)
    assert raw["C1"] - raw["C0"] < -0.3
    assert abs(out["differences"]["C1-C0"]["mean"]) < 0.05
    assert out["log_odds_per_unit"]["mean"] == pytest.approx(1.0, abs=0.15)  # per 10 words


def test_controlled_rates_cis_are_reproducible():
    data = _rows({"C0": 0.5, "C1": 0.2}, delta=lambda j: j - 5)
    a = controlled_win_rates(data, ["C0", "C1"], n_boot=30, seed=7)
    b = controlled_win_rates(data, ["C0", "C1"], n_boot=30, seed=7)
    assert a == b
