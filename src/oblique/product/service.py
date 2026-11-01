"""The flow behind the page: limits, a sequential set under three cards, visible sources.

No judges and no metrics: the same generation core as the experiment, on the user's documents.
Sets are generated with the Standard model, never Contributor, which would let Meta use the
user's PDFs for training.

The session limit counts sets per browser cookie, so clearing cookies resets it. The budget in
the persistent ledger is what really protects the credits.

The local response cache keeps every response, as in the rest of the project, but its copy of the
request is reduced to the model name, so the PDFs are never written to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oblique.cache import Cache
from oblique.deck import Card, load_deck
from oblique.deck.controls import DECK_DIR
from oblique.generate.schema import IdeaCard
from oblique.generate.validate import parse_set, unfence
from oblique.llm import Completion, ContentFilteredError, LLMClient
from oblique.product.documents import Document, normalize
from oblique.prompts.product import IDEATION_PDF_V1, pdf_ideation_messages

logger = logging.getLogger(__name__)

K = 3
MAX_JOBS_KEPT = 500


class LimitReached(Exception):
    """A usage limit. The message is safe to show to the user."""


class Busy(Exception):
    """Too many sets are being generated right now."""


class DocumentCache(Cache):
    """Stores the response under the same key, without the request and the PDFs inside it."""

    def put(self, key: str, *, model: str, request: dict[str, Any], **kw: Any) -> None:
        super().put(key, model=model, request={"model": model, "stored": False}, **kw)


def parse_sources(content: str | None, documents: list[Document]) -> dict[str, Any]:
    """The sources of a K = 1 reply. A source naming no uploaded document is dropped and counted;
    `found` says whether the quote is in the text extracted locally from that PDF. A quote that is
    not found may still be real (scanned pages, odd extraction), so the page shows it as unchecked,
    not as false."""
    try:
        items = json.loads(unfence(content or "")[0])["ideas"][0]["sources"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {"sources": [], "rejected": 0}
    if not isinstance(items, list):
        return {"sources": [], "rejected": 0}
    by_number = {d.number: d for d in documents}
    sources, rejected = [], 0
    for item in items:
        number = item.get("document") if isinstance(item, dict) else None
        quote = item.get("quote") if isinstance(item, dict) else None
        if isinstance(number, bool) or number not in by_number or not isinstance(quote, str):
            rejected += 1
            continue
        document = by_number[number]
        found = bool(quote.strip()) and normalize(quote) in document.text
        sources.append({"document": number, "name": document.name, "quote": quote, "found": found})
    return {"sources": sources, "rejected": rejected}


@dataclass
class Session:
    sets_started: int = 0
    spent_usd: float = 0.0


@dataclass
class Job:
    id: str
    session: str
    cards: list[Card]
    documents: list[dict[str, Any]]
    status: str = "running"  # running | done | limit | failed
    ideas: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    task: asyncio.Task | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "documents": self.documents,
            "k": len(self.cards),
            "ideas": self.ideas,
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
            concurrency=self.limits["max_running_sets"] * 2,
            max_attempts=gen["max_attempts"],
            timeout=gen["timeout_seconds"],
            client=transport,
            wait=wait,
        )
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, Job] = {}
        self.ledger = Path(cfg["ledger"])
        self.spent_usd = 0.0
        if self.ledger.exists():
            lines = self.ledger.read_text("utf-8").splitlines()
            self.spent_usd = sum(json.loads(line)["cost_usd"] for line in lines)

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
            "deck_note": self.cfg["deck"]["note"],
        }

    # --- a set ----------------------------------------------------------------

    def start(self, session_id: str, documents: list[Document]) -> Job:
        """Everything before the first await, so concurrent requests cannot pass a limit."""
        session = self.session(session_id)
        self.check_can_start(session)
        session.sets_started += 1
        job = Job(
            id=secrets.token_urlsafe(16),
            session=session_id,
            cards=random.SystemRandom().sample(self.deck, K),
            documents=[{"number": d.number, "name": d.name, "pages": d.pages} for d in documents],
        )
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if j.status != "running"][:-MAX_JOBS_KEPT]:
            del self.jobs[old.id]
        job.task = asyncio.create_task(self.run(job, session, documents))
        return job

    async def call(
        self, session: Session, messages: list[dict[str, Any]], attempt: int
    ) -> Completion:
        self.check_budget(session)
        c = await self.client.complete(
            self.gen["model"], messages, params=self.gen["params"], attempt=attempt
        )
        if not c.from_cache:
            u = c.usage
            cost = (
                (u.tokens_in - u.tokens_cached) * self.price["in"]
                + u.tokens_cached * self.price["cached"]
                + u.tokens_out * self.price["out"]
            ) / 1e6
            self.spent_usd += cost
            session.spent_usd += cost
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8", newline="\n") as f:
                row = {"key": c.key, "cost_usd": cost, "prompt_hash": IDEATION_PDF_V1.hash}
                f.write(json.dumps(row) + "\n")
        return c

    async def one_idea(
        self,
        session: Session,
        documents: list[Document],
        previous: list[IdeaCard],
        card: Card,
    ) -> tuple[dict[str, Any], IdeaCard | None]:
        shown = {"id": card.id, "text": card.text}
        messages = pdf_ideation_messages([d.data for d in documents], previous, card.text)
        try:
            c = await self.call(session, messages, 0)
            parsed = parse_set(c.content, 1)
            if parsed.retryable:  # ADR-006: one retry, json_error only
                c = await self.call(session, messages, 1)
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
                **parse_sources(c.content, documents),
            },
            slot.idea,
        )

    async def run(self, job: Job, session: Session, documents: list[Document]) -> None:
        previous: list[IdeaCard] = []
        try:
            for card in job.cards:
                shown, idea = await self.one_idea(session, documents, previous, card)
                job.ideas.append(shown)
                if idea is not None:
                    previous = [*previous, idea]
            job.status = "done"
        except LimitReached as e:
            job.status, job.message = "limit", str(e)
        except Exception:
            logger.exception("set %s failed", job.id)
            job.status, job.message = "failed", "The model call failed. Please try again."
            if not job.ideas:
                session.sets_started -= 1  # nothing delivered: it does not count against the user
