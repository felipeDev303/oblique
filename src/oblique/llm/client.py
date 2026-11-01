"""Single exit door for every model call (Meta Model API and Baseten).

Both providers speak the OpenAI Chat Completions protocol. Every call goes
through the local cache; only successful responses are cached. Transient
errors (429, timeouts, connection errors, 5xx) are retried with exponential
backoff and full jitter. Identical requests in flight are deduplicated so the
same response is never paid for twice. A request the provider's content filter
refuses gets one identical retry, then raises `ContentFilteredError` (ADR-006,
amendment 6): it is never cached, so a resumed run asks again.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import openai
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.wait import wait_base

from oblique.cache import Cache, make_key

logger = logging.getLogger(__name__)

META_BASE_URL = "https://api.meta.ai/v1"
SPARK_OPEN_BASETEN = "meta-models/muse-spark-open-1026"
CONTENT_FILTER = "content management policy"  # in the provider's 400 message
DEFAULT_TIMEOUT = 180.0  # seconds per attempt; generations take ~45 s (R-20261030-02)

RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class ContentFilteredError(Exception):
    """The provider's content filter refused a request twice. `key` is its cache key."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class Provider:
    name: str
    api_key_env: str
    base_url: str | None = None
    base_url_env: str | None = None

    def resolve(self) -> tuple[str, str]:
        base_url = self.base_url or os.environ.get(self.base_url_env or "", "")
        api_key = os.environ.get(self.api_key_env, "")
        missing = [
            n
            for n, v in ((self.base_url_env, base_url), (self.api_key_env, api_key))
            if n and not v
        ]
        if missing:
            raise RuntimeError(f"{self.name}: missing environment variables {missing}")
        return base_url, api_key


PROVIDERS = {
    "meta": Provider("meta", api_key_env="MODEL_API_KEY", base_url=META_BASE_URL),
    "baseten": Provider("baseten", api_key_env="BASETEN_API_KEY", base_url_env="BASETEN_BASE_URL"),
}


@dataclass(frozen=True)
class Usage:
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> Usage:
        u = response.get("usage") or {}
        prompt_details = u.get("prompt_tokens_details") or {}
        completion_details = u.get("completion_tokens_details") or {}
        return cls(
            tokens_in=u.get("prompt_tokens") or 0,
            tokens_cached=prompt_details.get("cached_tokens") or 0,
            tokens_out=u.get("completion_tokens") or 0,
            tokens_reasoning=completion_details.get("reasoning_tokens") or 0,
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.tokens_in + other.tokens_in,
            self.tokens_cached + other.tokens_cached,
            self.tokens_out + other.tokens_out,
            self.tokens_reasoning + other.tokens_reasoning,
        )


@dataclass(frozen=True)
class Completion:
    key: str
    model: str
    content: str | None
    reasoning: str | None
    usage: Usage
    from_cache: bool
    response: dict[str, Any]


@dataclass
class Stats:
    """Counters for this client's lifetime. `usage` counts only paid (non-cached) calls."""

    api_calls: int = 0
    cache_hits: int = 0
    deduplicated: int = 0
    retries: int = 0
    rate_limited: int = 0
    limit_reductions: int = 0
    content_filter_retries: int = 0
    content_filtered: int = 0
    usage: Usage = field(default_factory=Usage)


class ConcurrencyLimiter:
    """Bounds calls in flight. If `adaptive`, a 429 halves the limit (floor 1), and with a
    `max_limit` every `raise_after` straight successes add 1, up to that ceiling.

    `epoch` counts reductions. Calls started before a reduction cannot reduce again, so
    one burst of 429s from calls already in flight lowers the limit only once.
    """

    def __init__(
        self,
        limit: int,
        *,
        adaptive: bool = False,
        max_limit: int | None = None,
        raise_after: int = 50,
    ):
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if max_limit is not None and max_limit < limit:
            raise ValueError("max_limit must be >= limit")
        self.limit = limit
        self.adaptive = adaptive
        self.max_limit = max_limit
        self.raise_after = raise_after
        self.successes = 0
        self.epoch = 0
        self.active = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> ConcurrencyLimiter:
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < self.limit)
            self.active += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()

    def on_rate_limited(self, epoch: int) -> bool:
        """Report a 429 from a call started at `epoch`. True if the limit went down."""
        self.successes = 0
        if not self.adaptive or epoch != self.epoch or self.limit == 1:
            return False
        self.limit = max(1, self.limit // 2)
        self.epoch += 1
        return True

    def on_success(self) -> bool:
        """Report a successful call. True if the limit went up. Waiting calls see the new
        limit when this call leaves the limiter, which notifies them."""
        if not self.adaptive or self.max_limit is None or self.limit >= self.max_limit:
            return False
        self.successes += 1
        if self.successes < self.raise_after:
            return False
        self.successes = 0
        self.limit += 1
        return True


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or [{}]
    return choices[0].get("message") or {}


class LLMClient:
    def __init__(
        self,
        provider: str,
        cache: Cache,
        *,
        concurrency: int = 8,
        adaptive: bool = False,
        max_concurrency: int | None = None,
        max_attempts: int = 6,
        wait: wait_base | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: Any = None,
    ):
        self.provider = PROVIDERS[provider]
        self.cache = cache
        self.max_attempts = max_attempts
        self.wait = wait or wait_random_exponential(multiplier=1, max=60)
        self.timeout = timeout
        self.stats = Stats()
        self._client = client
        self.limiter = ConcurrencyLimiter(concurrency, adaptive=adaptive, max_limit=max_concurrency)
        self._inflight: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def _openai(self) -> Any:
        if self._client is None:
            base_url, api_key = self.provider.resolve()
            # Retries are handled here, not by the SDK.
            self._client = openai.AsyncOpenAI(
                base_url=base_url, api_key=api_key, max_retries=0, timeout=self.timeout
            )
        return self._client

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        params: dict[str, Any] | None = None,
        sample_idx: int = 0,
        attempt: int = 0,
    ) -> Completion:
        """One chat completion, served from the cache when possible.

        `params` go to the API verbatim (temperature, max_tokens, seed, ...)
        and are part of the cache key, as are the provider, `sample_idx` and `attempt`.
        """
        params = dict(params or {})
        key = make_key(model, messages, params, sample_idx, attempt, provider=self.provider.name)

        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return self._wrap(key, model, cached, from_cache=True)

        future = self._inflight.get(key)
        if future is None:
            future = asyncio.ensure_future(self._fetch(key, model, messages, params))
            self._inflight[key] = future
            future.add_done_callback(lambda _: self._inflight.pop(key, None))
        else:
            self.stats.deduplicated += 1
        # shield: a cancelled caller must not cancel the call other callers wait on.
        response = await asyncio.shield(future)
        return self._wrap(key, model, response, from_cache=False)

    async def _fetch(
        self, key: str, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        request = {"model": model, "messages": messages, **params}
        async with self.limiter:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_attempts),
                wait=self.wait,
                retry=retry_if_exception_type(RETRYABLE),
                before_sleep=self._on_retry,
                reraise=True,
            ):
                with attempt:
                    epoch = self.limiter.epoch
                    try:
                        raw = await self._create(key, request)
                    except openai.RateLimitError:
                        if self.limiter.on_rate_limited(epoch):
                            self.stats.limit_reductions += 1
                            logger.warning(
                                "%s: 429, concurrency limit lowered to %d",
                                self.provider.name,
                                self.limiter.limit,
                            )
                        raise
            self.limiter.on_success()
        response = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
        usage = Usage.from_response(response)
        self.cache.put(
            key,
            model=model,
            request=request,
            response=response,
            tokens_in=usage.tokens_in,
            tokens_cached=usage.tokens_cached,
            tokens_out=usage.tokens_out,
            tokens_reasoning=usage.tokens_reasoning,
        )
        self.stats.api_calls += 1
        self.stats.usage = self.stats.usage + usage
        return response

    async def _create(self, key: str, request: dict[str, Any]) -> Any:
        create = self._openai().chat.completions.create
        try:
            return await create(**request)
        except openai.BadRequestError as e:
            if CONTENT_FILTER not in str(e):
                raise
            self.stats.content_filter_retries += 1
        try:
            return await create(**request)
        except openai.BadRequestError as e:
            if CONTENT_FILTER not in str(e):
                raise
            self.stats.content_filtered += 1
            raise ContentFilteredError(key, str(e)) from e

    def _on_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        self.stats.retries += 1
        if isinstance(exc, openai.RateLimitError):
            self.stats.rate_limited += 1
        logger.warning(
            "%s: attempt %d failed with %s; retrying",
            self.provider.name,
            state.attempt_number,
            type(exc).__name__,
        )

    @staticmethod
    def _wrap(key: str, model: str, response: dict[str, Any], *, from_cache: bool) -> Completion:
        msg = _message(response)
        # Spark Open on Baseten returns reasoning separately from the answer.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return Completion(
            key=key,
            model=model,
            content=msg.get("content"),
            reasoning=reasoning,
            usage=Usage.from_response(response),
            from_cache=from_cache,
            response=response,
        )
