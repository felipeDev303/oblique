"""Seeds for every random process, derived from one master seed fixed before any result.

The master seed is the kickoff date. Each purpose gets its own child stream via
`SeedSequence.spawn`, so the purposes are independent and adding a draw in one
never shifts another. The order of `PURPOSES` is part of the pre-registration:
append new purposes, never reorder.
"""

from __future__ import annotations

import numpy as np

MASTER_SEED = 20261026
PURPOSES = ("splits", "cards", "bootstrap")


def seed_sequence(purpose: str, master: int = MASTER_SEED) -> np.random.SeedSequence:
    if purpose not in PURPOSES:
        raise KeyError(f"unknown seed purpose {purpose!r}; expected one of {PURPOSES}")
    return np.random.SeedSequence(master).spawn(len(PURPOSES))[PURPOSES.index(purpose)]


def rng(purpose: str, master: int = MASTER_SEED) -> np.random.Generator:
    """A fresh generator for `purpose`. Same purpose and master -> same stream."""
    return np.random.default_rng(seed_sequence(purpose, master))
