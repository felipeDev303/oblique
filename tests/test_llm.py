"""LLM client tests. A fake transport stands in for the API: nothing here makes a network call."""

import asyncio

import httpx
import openai
import pytest
from tenacity import wait_none

from oblique.cache import Cache
from oblique.llm import ContentFilteredError, LLMClient, Usage
from oblique.llm.client import ConcurrencyLimiter

MSGS = [{"role": "user", "content": "propose an idea"}]


def _response(content: str = "ok", *, reasoning: str | None = None) -> dict:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "x",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
    }


class _Raw:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data


def _status_error(cls, status: int, message: str = "boom"):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


class FakeTransport:
    """Mimics `AsyncOpenAI().chat.completions.create`."""

    def __init__(self, script=None, delay: float = 0.0):
        self.script = list(script or [])
        self.delay = delay
        self.calls: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.chat = self
        self.completions = self

    async def create(self, **request):
        self.calls.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            item = self.script.pop(0) if self.script else _response(f"r{len(self.calls)}")
            if isinstance(item, Exception):
                raise item
            return _Raw(item)
        finally:
            self.active -= 1


@pytest.fixture
def cache(tmp_path):
    with Cache(tmp_path / "llm.db") as c:
        yield c


def _client(cache, transport, **kw):
    return LLMClient("meta", cache, client=transport, wait=wait_none(), **kw)


def test_second_identical_call_is_served_from_cache(cache):
    transport = FakeTransport([_response("first")])
    llm = _client(cache, transport)

    async def run():
        a = await llm.complete("muse-spark", MSGS, params={"temperature": 1.0})
        b = await llm.complete("muse-spark", MSGS, params={"temperature": 1.0})
        return a, b

    a, b = asyncio.run(run())
    assert len(transport.calls) == 1
    assert (a.content, a.from_cache) == ("first", False)
    assert (b.content, b.from_cache) == ("first", True)
    assert llm.stats.cache_hits == 1
    assert transport.calls[0] == {"model": "muse-spark", "messages": MSGS, "temperature": 1.0}


def test_cache_survives_a_new_client(cache):
    asyncio.run(_client(cache, FakeTransport()).complete("m", MSGS))
    transport = FakeTransport()
    result = asyncio.run(_client(cache, transport).complete("m", MSGS))
    assert result.from_cache
    assert transport.calls == []


def test_sample_idx_and_attempt_make_distinct_calls(cache):
    transport = FakeTransport()
    llm = _client(cache, transport)

    async def run():
        for s in range(3):
            await llm.complete("m", MSGS, sample_idx=s)
        await llm.complete("m", MSGS, sample_idx=0, attempt=1)

    asyncio.run(run())
    assert len(transport.calls) == 4


def test_concurrent_identical_requests_are_deduplicated(cache):
    transport = FakeTransport(delay=0.05)
    llm = _client(cache, transport)

    async def run():
        return await asyncio.gather(*(llm.complete("m", MSGS) for _ in range(5)))

    results = asyncio.run(run())
    assert len(transport.calls) == 1
    assert {r.content for r in results} == {"r1"}
    assert llm.stats.deduplicated == 4


def test_semaphore_bounds_concurrency(cache):
    transport = FakeTransport(delay=0.02)
    llm = _client(cache, transport, concurrency=3)

    async def run():
        await asyncio.gather(*(llm.complete("m", MSGS, sample_idx=i) for i in range(12)))

    asyncio.run(run())
    assert len(transport.calls) == 12
    assert transport.max_active == 3


def test_default_timeout_is_180_seconds(cache):
    assert LLMClient("meta", cache).timeout == 180.0


def test_fixed_limit_ignores_429s(cache):
    transport = FakeTransport([_status_error(openai.RateLimitError, 429), _response("ok")])
    llm = _client(cache, transport, concurrency=8)
    asyncio.run(llm.complete("m", MSGS))
    assert llm.stats.rate_limited == 1
    assert llm.limiter.limit == 8 and llm.stats.limit_reductions == 0


def test_adaptive_limit_halves_on_a_429_and_stays_down(cache):
    transport = FakeTransport([_status_error(openai.RateLimitError, 429), _response("ok")])
    llm = _client(cache, transport, concurrency=8, adaptive=True)

    async def run():
        await llm.complete("m", MSGS)
        await asyncio.gather(*(llm.complete("m", MSGS, sample_idx=i + 1) for i in range(8)))

    transport.delay = 0.01
    asyncio.run(run())
    assert llm.limiter.limit == 4 and llm.stats.limit_reductions == 1
    assert transport.max_active <= 4


def test_adaptive_limit_rises_to_its_ceiling_and_drops_on_a_429():
    limiter = ConcurrencyLimiter(8, adaptive=True, max_limit=12, raise_after=3)
    assert [limiter.on_success() for _ in range(3)] == [False, False, True]
    assert limiter.limit == 9
    for _ in range(50):
        limiter.on_success()
    assert limiter.limit == 12
    limiter.on_success()
    assert limiter.limit == 12
    assert limiter.on_rate_limited(limiter.epoch) is True and limiter.limit == 6
    # The count starts over after a 429.
    limiter.on_success(), limiter.on_success()
    limiter.on_rate_limited(-1)
    assert limiter.successes == 0
    assert limiter.limit == 6


def test_limit_without_a_ceiling_never_rises():
    limiter = ConcurrencyLimiter(8, adaptive=True)
    for _ in range(200):
        assert limiter.on_success() is False
    assert limiter.limit == 8
    with pytest.raises(ValueError):
        ConcurrencyLimiter(8, adaptive=True, max_limit=4)


def test_client_climbs_to_the_ceiling_with_successes(cache):
    transport = FakeTransport(delay=0.001)
    llm = _client(cache, transport, concurrency=2, adaptive=True, max_concurrency=4)
    llm.limiter.raise_after = 5

    async def run():
        await asyncio.gather(*(llm.complete("m", MSGS, sample_idx=i) for i in range(40)))

    asyncio.run(run())
    assert llm.limiter.limit == 4
    assert transport.max_active <= 4 and transport.max_active > 2


def test_one_burst_of_429s_lowers_the_limit_once():
    limiter = ConcurrencyLimiter(8, adaptive=True)
    started = limiter.epoch
    assert limiter.on_rate_limited(started) is True
    # Other calls of the same burst, started before the reduction, do not reduce again.
    assert limiter.on_rate_limited(started) is False
    assert limiter.limit == 4
    assert limiter.on_rate_limited(limiter.epoch) is True and limiter.limit == 2
    assert limiter.on_rate_limited(limiter.epoch) is True and limiter.limit == 1
    assert limiter.on_rate_limited(limiter.epoch) is False and limiter.limit == 1


FILTERED = "filtered due to the prompt triggering our content management policy"


def test_a_filtered_request_gets_one_identical_retry(cache):
    transport = FakeTransport(
        [_status_error(openai.BadRequestError, 400, FILTERED), _response("ok")]
    )
    llm = _client(cache, transport)
    result = asyncio.run(llm.complete("m", MSGS))
    assert result.content == "ok"
    assert len(transport.calls) == 2 and transport.calls[0] == transport.calls[1]
    assert llm.stats.content_filter_retries == 1 and llm.stats.content_filtered == 0


def test_a_filter_that_persists_raises_and_is_not_cached(cache):
    error = _status_error(openai.BadRequestError, 400, FILTERED)
    transport = FakeTransport([error, error, _response("later")])
    llm = _client(cache, transport)
    with pytest.raises(ContentFilteredError) as e:
        asyncio.run(llm.complete("m", MSGS))
    assert len(transport.calls) == 2 and llm.stats.content_filtered == 1
    assert e.value.key not in cache
    # A resumed run asks again.
    assert asyncio.run(llm.complete("m", MSGS)).content == "later"


def test_other_bad_requests_are_not_retried(cache):
    transport = FakeTransport([_status_error(openai.BadRequestError, 400, "bad parameter")])
    with pytest.raises(openai.BadRequestError):
        asyncio.run(_client(cache, transport).complete("m", MSGS))
    assert len(transport.calls) == 1


def test_transient_errors_are_retried_and_not_cached(cache):
    transport = FakeTransport(
        [
            _status_error(openai.RateLimitError, 429),
            _status_error(openai.InternalServerError, 503),
            _response("finally"),
        ]
    )
    llm = _client(cache, transport)
    result = asyncio.run(llm.complete("m", MSGS))
    assert result.content == "finally"
    assert len(transport.calls) == 3
    assert llm.stats.retries == 2
    assert llm.stats.rate_limited == 1
    assert len(cache) == 1


def test_transport_retries_keep_the_same_cache_key(cache):
    # ADR-006 amendment 3: 429/5xx/timeout retries are not an `attempt`.
    from oblique.cache import make_key

    transport = FakeTransport(
        [
            _status_error(openai.RateLimitError, 429),
            openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid")),
            _response("ok"),
        ]
    )
    llm = _client(cache, transport)
    result = asyncio.run(llm.complete("m", MSGS, params={"temperature": 1.0}, sample_idx=2))
    expected = make_key("m", MSGS, {"temperature": 1.0}, sample_idx=2, attempt=0, provider="meta")
    assert result.key == expected
    assert len(transport.calls) == 3
    assert all(call == transport.calls[0] for call in transport.calls)  # identical requests
    assert len(cache) == 1 and expected in cache


def test_gives_up_after_max_attempts_and_caches_nothing(cache):
    transport = FakeTransport([_status_error(openai.RateLimitError, 429)] * 3)
    llm = _client(cache, transport, max_attempts=3)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(llm.complete("m", MSGS))
    assert len(transport.calls) == 3
    assert len(cache) == 0


def test_non_retryable_errors_fail_immediately(cache):
    transport = FakeTransport([_status_error(openai.BadRequestError, 400)])
    llm = _client(cache, transport)
    with pytest.raises(openai.BadRequestError):
        asyncio.run(llm.complete("m", MSGS))
    assert len(transport.calls) == 1
    assert len(cache) == 0


def test_unparseable_model_output_is_still_cached(cache):
    transport = FakeTransport([_response("this is not json {")])
    llm = _client(cache, transport)
    asyncio.run(llm.complete("m", MSGS))
    assert len(cache) == 1


def test_tokens_are_recorded(cache):
    llm = _client(cache, FakeTransport())

    async def run():
        await llm.complete("m", MSGS, sample_idx=0)
        await llm.complete("m", MSGS, sample_idx=1)
        return await llm.complete("m", MSGS, sample_idx=0)  # cache hit: not paid again

    result = asyncio.run(run())
    assert result.usage == Usage(tokens_in=100, tokens_cached=80, tokens_out=20, tokens_reasoning=7)
    assert llm.stats.api_calls == 2
    assert llm.stats.usage == Usage(200, 160, 40, 14)


def test_reasoning_is_separated_from_content(cache):
    transport = FakeTransport([_response("answer", reasoning="thinking...")])
    result = asyncio.run(_client(cache, transport).complete("m", MSGS))
    assert result.content == "answer"
    assert result.reasoning == "thinking..."


def test_missing_credentials_raise_clear_error(cache, monkeypatch):
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_BASE_URL", raising=False)
    llm = LLMClient("baseten", cache)
    with pytest.raises(RuntimeError, match="BASETEN"):
        asyncio.run(llm.complete("m", MSGS))
