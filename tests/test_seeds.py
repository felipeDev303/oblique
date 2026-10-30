import pytest

from oblique.seeds import MASTER_SEED, PURPOSES, rng, seed_sequence


def test_master_seed_is_the_kickoff_date():
    assert MASTER_SEED == 20261026


def test_purpose_order_is_frozen():
    # Reordering would silently change every derived stream.
    assert PURPOSES == ("splits", "cards", "bootstrap")
    for i, purpose in enumerate(PURPOSES):
        ss = seed_sequence(purpose)
        assert ss.entropy == MASTER_SEED
        assert ss.spawn_key == (i,)


def test_same_purpose_same_stream():
    assert (
        rng("splits").integers(0, 10**9, 5).tolist() == rng("splits").integers(0, 10**9, 5).tolist()
    )


def test_purposes_are_independent():
    draws = {p: tuple(rng(p).integers(0, 10**9, 5)) for p in PURPOSES}
    assert len(set(draws.values())) == len(PURPOSES)


def test_unknown_purpose_raises():
    with pytest.raises(KeyError):
        seed_sequence("judges")
