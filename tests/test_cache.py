import threading

from oblique.cache import Cache, make_key

MSGS = [{"role": "user", "content": "hello"}]


def test_key_is_stable_and_order_insensitive_for_params():
    k1 = make_key("m", MSGS, {"temperature": 1.0, "max_tokens": 10})
    k2 = make_key("m", MSGS, {"max_tokens": 10, "temperature": 1.0})
    assert k1 == k2
    assert len(k1) == 64


def test_key_changes_with_every_component():
    base = make_key("m", MSGS, {"temperature": 1.0})
    assert make_key("other", MSGS, {"temperature": 1.0}) != base
    assert make_key("m", [{"role": "user", "content": "bye"}], {"temperature": 1.0}) != base
    assert make_key("m", MSGS, {"temperature": 0.0}) != base
    assert make_key("m", MSGS, {"temperature": 1.0}, sample_idx=1) != base
    assert make_key("m", MSGS, {"temperature": 1.0}, attempt=1) != base


def test_independent_samples_and_retries_do_not_collide():
    keys = {make_key("m", MSGS, {}, sample_idx=s, attempt=a) for s in range(3) for a in range(2)}
    assert len(keys) == 6


def test_put_get_roundtrip(tmp_path):
    with Cache(tmp_path / "c.db") as cache:
        key = make_key("m", MSGS, {})
        assert cache.get(key) is None
        assert key not in cache
        cache.put(key, model="m", request={"x": 1}, response={"text": "ñandú"}, tokens_in=5)
        assert cache.get(key) == {"text": "ñandú"}
        assert key in cache
        assert len(cache) == 1


def test_existing_entry_is_never_overwritten(tmp_path):
    with Cache(tmp_path / "c.db") as cache:
        cache.put("k", model="m", request={}, response={"v": 1})
        cache.put("k", model="m", request={}, response={"v": 2})
        assert cache.get("k") == {"v": 1}
        assert len(cache) == 1


def test_persists_across_connections(tmp_path):
    path = tmp_path / "nested" / "c.db"
    with Cache(path) as cache:
        cache.put("k", model="m", request={}, response={"v": 1})
    with Cache(path) as cache:
        assert cache.get("k") == {"v": 1}


def test_concurrent_writes_from_threads(tmp_path):
    with Cache(tmp_path / "c.db") as cache:

        def write(i: int) -> None:
            cache.put(f"k{i}", model="m", request={}, response={"i": i})

        threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(cache) == 50
        assert cache.get("k17") == {"i": 17}
