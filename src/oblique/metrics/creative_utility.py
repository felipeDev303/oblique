"""Creative Utility (Wadhwa et al., 2026), evaluation version with greedy ordering.

    CU(U) = sum_i gamma^(i-1) * q(u_i) * min_{j<i} d(u_i, u_j) / sum_i gamma^(i-1),
    with min_{j<1} d := 1.

The max over orderings is approximated greedily: first the idea with the
highest q, then repeatedly the one with the highest marginal value
q(u) * min_{chosen} d(u, chosen). The result is a lower bound on the true CU
and does not depend on the order in which the model wrote the ideas.

Ideas with q = 0 contribute nothing and are skipped as distance references.
This does not change the result (they are only picked once every remaining
marginal is already 0) and lets invalid ideas carry NaN distances.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

GAMMA = 0.9


def _check(q: Sequence[float], D: Sequence[Sequence[float]]) -> None:
    k = len(q)
    if k == 0:
        raise ValueError("empty idea set")
    if len(D) != k or any(len(row) != k for row in D):
        raise ValueError(f"D must be {k}x{k}")
    for i, qi in enumerate(q):
        if math.isnan(qi) or not 0.0 <= qi <= 1.0:
            raise ValueError(f"q[{i}] = {qi} outside [0, 1]")
    for i in range(k):
        for j in range(k):
            if i == j or q[i] == 0 or q[j] == 0:
                continue
            d = D[i][j]
            if math.isnan(d) or not 0.0 <= d <= 1.0:
                raise ValueError(f"D[{i}][{j}] = {d} outside [0, 1]")


def greedy_order(q: Sequence[float], D: Sequence[Sequence[float]]) -> tuple[list[int], list[float]]:
    """Greedy order and the marginal value of each idea at its position."""
    q = [float(x) for x in q]
    _check(q, D)
    remaining = list(range(len(q)))

    first = max(remaining, key=lambda i: q[i])
    order, marginals = [first], [q[first]]
    remaining.remove(first)

    def marginal(i: int) -> float:
        if q[i] == 0:
            return 0.0
        return q[i] * min(D[i][j] for j in order if q[j] > 0)

    while remaining:
        best = max(remaining, key=marginal)
        marginals.append(marginal(best))
        order.append(best)
        remaining.remove(best)
    return order, marginals


def creative_utility(
    q: Sequence[float], D: Sequence[Sequence[float]], gamma: float = GAMMA
) -> float:
    """CU in [0, 1] for one set. q: quality per idea; D: KxK distance matrix."""
    _, marginals = greedy_order(q, D)
    weights = [gamma**i for i in range(len(marginals))]
    return sum(w * m for w, m in zip(weights, marginals, strict=True)) / sum(weights)
