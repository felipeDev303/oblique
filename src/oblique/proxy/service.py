"""The proxy behind `POST /v1/chat/completions`: no judges, no streaming, Standard only.

Opt-in by model. `muse-spark-1.3+oblique` writes K answers in sequence, each under a different card
from the configured deck, and returns them in one choice, so a client that reads only the first
choice still sees all of them. The cards are in `oblique.cards_used`. `muse-spark-1.3` passes the
request through unchanged. Any other model is refused, so a Contributor model can never receive a
client's data.

Spend: the same ledger, budget and session limit as the app, taken from the app's config. The
client's API key is never forwarded; it only identifies the session, by its hash.

Like every call in this project, these go through the response cache, so an identical request can
return the same answer at no cost. The cache keeps responses but not the requests.
"""

from __future__ import annotations

import hashlib
import logging
import random
import secrets
import time
from pathlib import Path
from typing import Any

import openai
import yaml

from oblique.cache import Cache
from oblique.deck import Card, load_deck
from oblique.deck.controls import DECK_DIR
from oblique.llm import Completion, ContentFilteredError, LLMClient
from oblique.product.service import DocumentCache, LimitReached
from oblique.product.spend import SpendLedger, call_cost
from oblique.prompts.proxy import PROXY_SEQUENCE_V1, oblique_messages

logger = logging.getLogger(__name__)

OPT_IN_SUFFIX = "+oblique"
SEPARATOR = "\n\n---\n\n"
# What a set of answers can honour. Anything else (tools, response_format, ...) is refused, not
# dropped: the client would otherwise believe it took effect.
OBLIQUE_PARAMS = {
    "temperature",
    "top_p",
    "stop",
    "reasoning_effort",
    "max_tokens",
    "max_completion_tokens",
}


class ProxyError(Exception):
    """An error in the shape OpenAI clients expect. The message is safe to show."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        kind: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.body = {"error": {"message": message, "type": kind, "param": param, "code": code}}


def session_of(authorization: str | None) -> str:
    """A hash of the client's key: the key itself is neither stored nor forwarded."""
    return hashlib.sha256((authorization or "anonymous").encode()).hexdigest()[:24]


class ProxyService:
    def __init__(self, cfg: dict[str, Any], *, transport: Any = None, wait: Any = None):
        base = yaml.safe_load(Path(cfg["product_config"]).read_text("utf-8"))
        gen = base["generator"]
        if "contributor" in gen["model"]:
            raise SystemExit("the proxy must use the Standard model: requests are the user's data")
        self.model = gen["model"]
        self.opt_in_model = self.model + OPT_IN_SUFFIX
        self.limits = base["limits"]
        self.price = base["prices_per_m"][self.model]
        self.ledger = SpendLedger(base["ledger"])
        self.k = cfg["k"]
        self.max_tokens = cfg["max_tokens"]
        deck_cfg = cfg.get("deck") or base["deck"]
        deck = load_deck(cfg.get("deck_dir", DECK_DIR))
        self.deck: list[Card] = [deck[c] for c in deck_cfg["cards"]]
        self.deck_note = deck_cfg["note"]
        if len(self.deck) < self.k or any(c.origin == "control" for c in self.deck):
            raise SystemExit(f"the deck needs at least {self.k} cards and no controls")
        cache: Cache = DocumentCache(cfg["cache"])
        self.client = LLMClient(
            gen["provider"],
            cache,
            concurrency=cfg["concurrency"],
            max_attempts=gen["max_attempts"],
            timeout=gen["timeout_seconds"],
            client=transport,
            wait=wait,
        )
        self.session_spent: dict[str, float] = {}

    # --- limits ---------------------------------------------------------------

    def check_budget(self, session: str) -> None:
        if self.ledger.total() >= self.limits["budget_usd"]:
            raise LimitReached("This proxy has used its budget for now.")
        if self.session_spent.get(session, 0.0) >= self.limits["session_usd"]:
            raise LimitReached("This session has reached its spending limit.")

    # --- a call ---------------------------------------------------------------

    async def call(
        self,
        session: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        prompt_hash: str | None,
    ) -> Completion:
        self.check_budget(session)
        c = await self.client.complete(self.model, messages, params=params)
        if not c.from_cache:
            cost = call_cost(c.usage, self.price)
            self.session_spent[session] = self.session_spent.get(session, 0.0) + cost
            self.ledger.record(
                {
                    "key": c.key,
                    "kind": "proxy",
                    "prompt_hash": prompt_hash,
                    "tokens_in": c.usage.tokens_in,
                    "tokens_cached": c.usage.tokens_cached,
                    "tokens_out": c.usage.tokens_out,
                    "tokens_reasoning": c.usage.tokens_reasoning,
                    "cost_usd": cost,
                }
            )
        return c

    def bounded(self, params: dict[str, Any]) -> dict[str, Any]:
        """The client's tokens limit, never above the ceiling that protects the credits."""
        out = dict(params)
        asked = out.pop("max_completion_tokens", None)
        asked = out.pop("max_tokens", asked)
        if asked is None:
            out["max_tokens"] = self.max_tokens
        elif isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:
            raise ProxyError(400, "max_tokens must be a positive integer.", param="max_tokens")
        else:
            out["max_tokens"] = min(asked, self.max_tokens)
        return out

    # --- the request ----------------------------------------------------------

    def validate(self, body: dict[str, Any]) -> None:
        model = body.get("model")
        if model not in (self.model, self.opt_in_model):
            raise ProxyError(
                404,
                f"The model {model!r} is not served here. Use {self.model!r}, or "
                f"{self.opt_in_model!r} for a set of answers under oblique cards.",
                param="model",
                code="model_not_found",
            )
        if body.get("stream"):
            raise ProxyError(400, "Streaming is not supported.", param="stream")
        if body.get("n") not in (None, 1):
            raise ProxyError(400, "Only n = 1 is supported.", param="n")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ProxyError(400, "messages must be a non-empty list.", param="messages")
        if not all(isinstance(m, dict) and "role" in m for m in messages):
            raise ProxyError(400, "Each message must be an object with a role.", param="messages")

    async def complete(self, body: dict[str, Any], session: str) -> dict[str, Any]:
        self.validate(body)
        ignored = {"model", "messages", "stream", "n", "stream_options", "user"}
        params = {k: v for k, v in body.items() if k not in ignored}
        if body["model"] == self.opt_in_model:
            return await self.oblique_set(body["messages"], params, session)
        return await self.pass_through(body["messages"], params, session)

    async def guarded(self, session: str, messages, params, prompt_hash=None) -> Completion:
        """One provider call, with its errors in the shape clients expect."""
        try:
            return await self.call(session, messages, params, prompt_hash)
        except openai.BadRequestError as e:
            logger.warning("provider refused a request: %s", e)
            raise ProxyError(400, "The provider rejected the request.") from e
        except openai.APIError as e:
            logger.exception("provider error")
            raise ProxyError(502, "The model call failed.", kind="api_error") from e

    async def pass_through(
        self, messages: list[dict[str, Any]], params: dict[str, Any], session: str
    ) -> dict[str, Any]:
        try:
            c = await self.guarded(session, messages, self.bounded(params))
        except LimitReached as e:
            raise ProxyError(429, str(e), kind="insufficient_quota") from e
        except ContentFilteredError as e:
            raise self.filtered_error() from e
        return c.response

    def filtered_error(self) -> ProxyError:
        return ProxyError(
            400, "The provider's content filter refused the request.", code="content_filter"
        )

    async def oblique_set(
        self, messages: list[dict[str, Any]], params: dict[str, Any], session: str
    ) -> dict[str, Any]:
        unsupported = sorted(set(params) - OBLIQUE_PARAMS)
        if unsupported:
            raise ProxyError(
                400,
                f"{unsupported[0]} is not supported with {self.opt_in_model!r}.",
                param=unsupported[0],
            )
        last = messages[-1]
        if last["role"] != "user" or not isinstance(last.get("content"), str | list):
            raise ProxyError(400, "The last message must be from the user.", param="messages")
        params = self.bounded(params)
        cards = random.SystemRandom().sample(self.deck, self.k)
        answers: list[dict[str, Any]] = []
        texts: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        finish = "stop"
        stopped: str | None = None
        for card in cards:
            entry: dict[str, Any] = {"card": card.id}
            try:
                c = await self.guarded(
                    session,
                    oblique_messages(messages, texts, card.text),
                    params,
                    PROXY_SEQUENCE_V1.hash,
                )
            except LimitReached as e:
                if not texts:
                    raise ProxyError(429, str(e), kind="insufficient_quota") from e
                stopped = str(e)
                break
            except ContentFilteredError:
                answers.append(entry | {"status": "filtered"})
                continue
            usage["prompt_tokens"] += c.usage.tokens_in
            usage["completion_tokens"] += c.usage.tokens_out
            usage["total_tokens"] += c.usage.tokens_in + c.usage.tokens_out
            choice = (c.response.get("choices") or [{}])[0]
            if not (c.content or "").strip():
                answers.append(entry | {"status": "empty"})
                continue
            if choice.get("finish_reason") == "length":
                finish = "length"
            texts.append(c.content)
            answers.append(entry | {"status": "ok", "content": c.content})
        if not texts:
            raise self.filtered_error()
        used = {a["card"] for a in answers if a["status"] == "ok"}
        return {
            "id": f"chatcmpl-oblique-{secrets.token_urlsafe(16)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.opt_in_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": SEPARATOR.join(texts)},
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
            "oblique": {
                "k": self.k,
                "cards_used": [
                    {"id": card.id, "text": card.text} for card in cards if card.id in used
                ],
                "answers": answers,
                "stopped": stopped,
                "deck_note": self.deck_note,
            },
        }
