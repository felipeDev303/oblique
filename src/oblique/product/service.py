"""The flow behind the page: limits, one sheet per PDF, a sequential set under three cards.

Each PDF goes in its own call and becomes a sheet (title, summary, methods, limitations, quoted
evidence). The ideas are written over the sheets and never see a PDF, so the cost of a set grows
with one read per document, not with three reads of every document.

By default a set is sequential (ADR-009): K calls, each one under a card and reading the ideas
before it, because on split B the sequential set beat the one-call set clearly (C1s - C1, ADR-011).
The cards can be turned off. The side-by-side view adds one call that asks for the K ideas at once
with no card, as C1 in the experiment, and shows it next to the sequential set.

No judges and no metrics: the same generation core as the experiment, on the user's documents.
Sets are generated with the Standard model, never Contributor, which would let Meta use the
user's PDFs for training.

The session limit counts sets per browser cookie, so clearing cookies resets it. The budget in
the persistent ledger is what really protects the credits.

The local response cache keeps every response, sheets included, as in the rest of the project, but
its copy of the request is reduced to the model name, so the PDFs are never written to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
from dataclasses import dataclass, field
from typing import Any

from oblique.cache import Cache
from oblique.deck import Card, load_deck
from oblique.deck.controls import DECK_DIR
from oblique.generate.schema import IdeaCard
from oblique.generate.validate import parse_set, unfence
from oblique.llm import Completion, ContentFilteredError, LLMClient
from oblique.product.documents import Document, Sheet, parse_sheet
from oblique.product.spend import SpendLedger, call_cost
from oblique.prompts.product import (
    DOCUMENT_SHEET_V1,
    IDEATION_SHEETS_ONE_CALL_V1,
    IDEATION_SHEETS_V1,
    document_sheet_messages,
    idea_messages,
    one_call_messages,
)

logger = logging.getLogger(__name__)

K = 3
MAX_JOBS_KEPT = 500
VIEWS = ("sequential", "side_by_side")
PROMPTS = {
    "sheet": DOCUMENT_SHEET_V1,
    "idea": IDEATION_SHEETS_V1,
    "one_call": IDEATION_SHEETS_ONE_CALL_V1,
}


class LimitReached(Exception):
    """A usage limit. The message is safe to show to the user."""


class Busy(Exception):
    """Too many sets are being generated right now."""


class SheetFailed(Exception):
    """A document that produced no sheet. The message is safe to show to the user."""


class DocumentCache(Cache):
    """Stores the response under the same key, without the request and the PDFs inside it."""

    def put(self, key: str, *, model: str, request: dict[str, Any], **kw: Any) -> None:
        super().put(key, model=model, request={"model": model, "stored": False}, **kw)


def parse_sources(content: str | None, sheets: list[Sheet], index: int = 0) -> dict[str, Any]:
    """The sources of idea `index` of a reply, resolved to the evidence of the sheets. A source that
    names no such document or passage is dropped and counted: only a passage from a sheet can be
    shown."""
    try:
        items = json.loads(unfence(content or "")[0])["ideas"][index]["sources"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {"sources": [], "rejected": 0}
    if not isinstance(items, list):
        return {"sources": [], "rejected": 0}
    by_number = {s.number: s for s in sheets}
    sources, rejected = [], 0
    for item in items:
        number = item.get("document") if isinstance(item, dict) else None
        evidence = item.get("evidence") if isinstance(item, dict) else None
        sheet = None
        if not isinstance(number, bool) and isinstance(number, int):
            sheet = by_number.get(number)
        passage = next((e for e in sheet.evidence if e.id == evidence), None) if sheet else None
        if passage is None:
            rejected += 1
            continue
        sources.append(
            {
                "document": number,
                "name": sheet.name,
                "evidence": passage.id,
                "quote": passage.quote,
                "found": passage.found,
            }
        )
    return {"sources": sources, "rejected": rejected}


@dataclass
class Session:
    sets_started: int = 0
    spent_usd: float = 0.0


@dataclass
class Job:
    id: str
    session: str
    cards: list[Card | None]  # None in every slot when the cards are off
    documents: list[dict[str, Any]]
    view: str = "sequential"  # sequential | side_by_side
    status: str = "running"  # running | done | limit | failed
    phase: str = "reading"  # reading | one_call | ideas
    sheets: list[dict[str, Any]] = field(default_factory=list)
    ideas: list[dict[str, Any]] = field(default_factory=list)  # the sequential set
    one_call: list[dict[str, Any]] = field(default_factory=list)  # side by side only
    message: str | None = None
    task: asyncio.Task | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "phase": self.phase,
            "message": self.message,
            "documents": self.documents,
            "k": len(self.cards),
            "view": self.view,
            "with_cards": any(c is not None for c in self.cards),
            "sheets": self.sheets,
            "ideas": self.ideas,
            "one_call": self.one_call,
        }


class ProductService:
    def __init__(self, cfg: dict[str, Any], *, transport: Any = None, wait: Any = None):
        gen = cfg["generator"]
        if "contributor" in gen["model"]:
            raise SystemExit("the product must use the Standard model: PDFs are the user's data")
        self.cfg = cfg
        self.gen = gen
        self.limits = cfg["limits"]
        self.price = cfg["prices_per_m"][gen["model"]]
        deck = load_deck(cfg.get("deck_dir", DECK_DIR))
        self.deck = [deck[c] for c in cfg["deck"]["cards"]]
        if len(self.deck) < K or any(c.origin == "control" for c in self.deck):
            raise SystemExit(f"the deck needs at least {K} cards and no controls")
        self.client = LLMClient(
            gen["provider"],
            DocumentCache(cfg["cache"]),
            concurrency=self.limits["max_running_sets"],
            max_attempts=gen["max_attempts"],
            timeout=gen["timeout_seconds"],
            client=transport,
            wait=wait,
        )
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, Job] = {}
        self.ledger = SpendLedger(cfg["ledger"])

    @property
    def spent_usd(self) -> float:
        return self.ledger.total()

    # --- limits ---------------------------------------------------------------

    def session(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session())

    def check_budget(self, session: Session) -> None:
        if self.spent_usd >= self.limits["budget_usd"]:
            raise LimitReached("This demo has used its budget for now.")
        if session.spent_usd >= self.limits["session_usd"]:
            raise LimitReached("This session has reached its spending limit.")

    def check_can_start(self, session: Session) -> None:
        if session.sets_started >= self.limits["sets_per_session"]:
            raise LimitReached(f"This session has used its {self.limits['sets_per_session']} sets.")
        self.check_budget(session)
        running = sum(j.status == "running" for j in self.jobs.values())
        if running >= self.limits["max_running_sets"]:
            raise Busy("Too many sets are being generated right now. Try again in a minute.")

    def status(self, session_id: str) -> dict[str, Any]:
        session = self.session(session_id)
        used = session.sets_started
        return {
            "sets_left": max(0, self.limits["sets_per_session"] - used),
            "min_documents": self.limits["min_documents"],
            "max_documents": self.limits["max_documents"],
            "max_file_mb": self.limits["max_file_mb"],
            "max_pages_per_document": self.limits["max_pages_per_document"],
            "model": self.gen["model"],
            "deck": self.cfg["deck"].get("name"),
            "deck_experimental": bool(self.cfg["deck"].get("experimental", False)),
            "deck_note": self.cfg["deck"]["note"],
        }

    # --- a set ----------------------------------------------------------------

    def start(
        self,
        session_id: str,
        documents: list[Document],
        *,
        view: str = "sequential",
        with_cards: bool = True,
    ) -> Job:
        """Everything before the first await, so concurrent requests cannot pass a limit."""
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}")
        session = self.session(session_id)
        self.check_can_start(session)
        session.sets_started += 1
        job = Job(
            id=secrets.token_urlsafe(16),
            session=session_id,
            cards=random.SystemRandom().sample(self.deck, K) if with_cards else [None] * K,
            documents=[{"number": d.number, "name": d.name, "pages": d.pages} for d in documents],
            view=view,
        )
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if j.status != "running"][:-MAX_JOBS_KEPT]:
            del self.jobs[old.id]
        job.task = asyncio.create_task(self.run(job, session, documents))
        return job

    async def call(
        self,
        session: Session,
        kind: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        attempt: int,
    ) -> Completion:
        self.check_budget(session)
        c = await self.client.complete(self.gen["model"], messages, params=params, attempt=attempt)
        if not c.from_cache:
            u = c.usage
            cost = call_cost(u, self.price)
            session.spent_usd += cost
            row = {
                "key": c.key,
                "kind": kind,
                "prompt_hash": PROMPTS[kind].hash,
                "tokens_in": u.tokens_in,
                "tokens_cached": u.tokens_cached,
                "tokens_out": u.tokens_out,
                "tokens_reasoning": u.tokens_reasoning,
                "cost_usd": cost,
            }
            self.ledger.record(row)
        return c

    async def read_document(self, session: Session, document: Document) -> Sheet:
        """One PDF, one call (one retry if the reply breaks the sheet's contract)."""
        messages = document_sheet_messages(document.data)
        params = self.gen["sheet_params"]
        try:
            c = await self.call(session, "sheet", messages, params, 0)
            sheet = parse_sheet(c.content, document)
            if sheet is None:
                c = await self.call(session, "sheet", messages, params, 1)
                sheet = parse_sheet(c.content, document)
        except ContentFilteredError as e:
            raise SheetFailed(f"The provider's content filter refused {document.name}.") from e
        if sheet is None:
            raise SheetFailed(f"{document.name} could not be read into a sheet.")
        return sheet

    async def read_all(self, session: Session, documents: list[Document]) -> list[Sheet]:
        """One document at a time: the limits are checked before each call, so a set can pass one
        only by the cost of the sheet in flight. In parallel, 5 PDFs could all pass it together."""
        return [await self.read_document(session, d) for d in documents]

    async def one_idea(
        self,
        session: Session,
        sheets: list[Sheet],
        previous: list[IdeaCard],
        card: Card | None,
    ) -> tuple[dict[str, Any], IdeaCard | None]:
        shown = {"id": card.id, "text": card.text} if card else None
        messages = idea_messages(sheets, previous, card.text if card else None)
        params = self.gen["params"]
        try:
            c = await self.call(session, "idea", messages, params, 0)
            parsed = parse_set(c.content, 1)
            if parsed.retryable:  # ADR-006: one retry, json_error only
                c = await self.call(session, "idea", messages, params, 1)
                parsed = parse_set(c.content, 1)
        except ContentFilteredError:
            return {"card": shown, "status": "filtered"}, None
        slot = parsed.slots[0]
        if slot.idea is None:
            return {"card": shown, "status": "invalid"}, None
        return (
            {
                "card": shown,
                "status": "ok",
                "idea": slot.idea.model_dump(),
                **parse_sources(c.content, sheets),
            },
            slot.idea,
        )

    async def one_call(self, session: Session, sheets: list[Sheet]) -> list[dict[str, Any]]:
        """The K ideas in one call, with no card. A broken set fails all K ideas (ADR-006)."""
        messages = one_call_messages(sheets, K)
        params = self.gen["params"]
        try:
            c = await self.call(session, "one_call", messages, params, 0)
            parsed = parse_set(c.content, K)
            if parsed.retryable:  # ADR-006: one retry, json_error only
                c = await self.call(session, "one_call", messages, params, 1)
                parsed = parse_set(c.content, K)
        except ContentFilteredError:
            return [{"card": None, "status": "filtered"} for _ in range(K)]
        out = []
        for i, slot in enumerate(parsed.slots):
            if slot.idea is None:
                out.append({"card": None, "status": "invalid"})
                continue
            out.append(
                {
                    "card": None,
                    "status": "ok",
                    "idea": slot.idea.model_dump(),
                    **parse_sources(c.content, sheets, i),
                }
            )
        return out

    async def run(self, job: Job, session: Session, documents: list[Document]) -> None:
        previous: list[IdeaCard] = []
        try:
            sheets = await self.read_all(session, documents)
            job.sheets = [s.public() for s in sheets]
            if job.view == "side_by_side":
                job.phase = "one_call"
                job.one_call = await self.one_call(session, sheets)
            job.phase = "ideas"
            for card in job.cards:
                shown, idea = await self.one_idea(session, sheets, previous, card)
                job.ideas.append(shown)
                if idea is not None:
                    previous = [*previous, idea]
            job.status = "done"
        except LimitReached as e:
            job.status, job.message = "limit", str(e)
        except SheetFailed as e:
            job.status, job.message = "failed", str(e)
            session.sets_started -= 1  # nothing delivered: it does not count against the user
        except Exception:
            logger.exception("set %s failed", job.id)
            job.status, job.message = "failed", "The model call failed. Please try again."
            if not job.ideas and not job.one_call:
                session.sets_started -= 1  # nothing delivered: it does not count against the user
