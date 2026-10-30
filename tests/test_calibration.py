import numpy as np
import pytest

from oblique.judges.embed import field_distance, pair_distance
from oblique.metrics.calibration import (
    agreement,
    cohen_kappa,
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
