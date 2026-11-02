"""Bring your papers: limits, sheets, the sequential set, sources and privacy. No API calls."""

import asyncio
import base64
import json
import sqlite3
import time
from pathlib import Path

import httpx
import openai
import pytest
import yaml
from fastapi.testclient import TestClient
from tenacity import wait_none

from oblique.deck import load_deck
from oblique.deck.controls import DECK_DIR
from oblique.generate.schema import IdeaCard
from oblique.product.app import COOKIE, create_app
from oblique.product.documents import (
    UploadError,
    clean_name,
    load_documents,
    normalize,
    parse_sheet,
)
from oblique.product.service import ProductService, parse_sources
from oblique.prompts.product import (
    IDEA_SYSTEM,
    SHEET_SYSTEM,
    document_sheet_messages,
    idea_messages,
)

LIMITS = {
    "min_documents": 2,
    "max_documents": 5,
    "max_file_mb": 1,
    "max_pages_per_document": 3,
    "sets_per_session": 2,
    "session_usd": 1.0,
    "budget_usd": 10.0,
    "max_running_sets": 2,
}


def make_pdf(text: str, pages: int = 1) -> bytes:
    """A small real PDF with one line of text per page."""
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    ]
    font = 3 + 2 * pages
    for i in range(pages):
        stream = f"BT /F1 12 Tf 20 700 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    start = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        start,
    )
    return bytes(out)


PDF_A = make_pdf("Alpha measured seventeen samples in the cold room")
PDF_B = make_pdf("Beta reported a failure under load")
PAIR = [("a.pdf", PDF_A), ("b.pdf", PDF_B)]
DOCUMENTS = load_documents(PAIR, LIMITS)


def sheet_reply(quotes=("measured seventeen samples", "a sentence the paper never wrote")):
    return json.dumps(
        {
            "title": "A title",
            "summary": "A summary.",
            "methods": "Some methods.",
            "limitations": "Some limits.",
            "evidence": list(quotes),
        }
    )


SHEETS = [parse_sheet(sheet_reply(), d) for d in DOCUMENTS]


def idea_reply(tag="x", sources=None):
    sources = [{"document": 1, "evidence": "E1"}] if sources is None else sources
    idea = {
        "problem_gap": f"[Evidence Gap] gap {tag}",
        "core_claim": f"claim {tag}",
        "method": "[Empirical Mapping] method",
        "inspiration": "inspiration",
        "sources": sources,
    }
    return json.dumps({"ideas": [idea]})


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def filtered():
    request = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError("content management policy", response=response, body=None)


class FakeAPI:
    """`sheet` and `idea` take the number of the call of that kind and may return text, None for
    the default reply, or an exception to raise."""

    def __init__(self, sheet=None, idea=None, tokens=(1000, 0, 500), one_call=None):
        self.chat = self.completions = self
        self.requests = []
        self.sheet = sheet
        self.idea = idea
        self.one_call = one_call
        self.tokens = tokens

    def of(self, system):
        return [r for r in self.requests if r["messages"][0]["content"] == system]

    def one_calls(self):
        return [
            r for r in self.of(IDEA_SYSTEM) if "exactly 3 objects" in r["messages"][1]["content"]
        ]

    async def create(self, **request):
        self.requests.append(request)
        if request["messages"][0]["content"] == SHEET_SYSTEM:
            n = len(self.of(SHEET_SYSTEM))
            content = self.sheet(n) if self.sheet else None
            content = sheet_reply() if content is None else content
        elif "exactly 3 objects" in request["messages"][1]["content"]:
            content = self.one_call or json.dumps(
                {"ideas": [json.loads(idea_reply(f"one-{i}"))["ideas"][0] for i in range(3)]}
            )
        else:
            n = len(self.of(IDEA_SYSTEM)) - len(self.one_calls())
            content = self.idea(n) if self.idea else None
            content = idea_reply(str(n)) if content is None else content
        if isinstance(content, Exception):
            raise content
        prompt, cached, completion = self.tokens
        return _Raw(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        )


@pytest.fixture
def cfg(tmp_path):
    return {
        "cache": str(tmp_path / "cache.db"),
        "ledger": str(tmp_path / "ledger.jsonl"),
        "page": str(Path("web/product/index.html")),
        "generator": {
            "provider": "meta",
            "model": "muse-spark-1.3",
            "params": {"temperature": 1.0, "response_format": {"type": "json_object"}},
            "sheet_params": {"temperature": 0, "response_format": {"type": "json_object"}},
            "timeout_seconds": 10,
            "max_attempts": 1,
        },
        "prices_per_m": {"muse-spark-1.3": {"in": 1.25, "cached": 0.15, "out": 4.25}},
        "limits": LIMITS,
        "deck": {"note": "Provisional.", "cards": ["OB-001", "OB-004", "OB-005", "OB-007"]},
    }


def run_set(service, session="s", documents=DOCUMENTS, **kw):
    async def go():
        job = service.start(session, documents, **kw)
        await job.task
        return job

    return asyncio.run(go())


# --- uploads --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files, expected",
    [
        ([("a.pdf", PDF_A)], "between 2 and 5"),
        ([(f"{i}.pdf", make_pdf(f"text {i}")) for i in range(6)], "between 2 and 5"),
        ([("a.pdf", PDF_A), ("b.txt", b"hello")], "not a PDF"),
        ([("a.pdf", PDF_A), ("copy.pdf", PDF_A)], "uploaded twice"),
        ([("a.pdf", PDF_A), ("long.pdf", make_pdf("x", pages=4))], "4 pages; the limit is 3"),
        ([("a.pdf", PDF_A), ("big.pdf", b"%PDF-" + b"0" * 1_000_001)], "larger than 1 MB"),
        ([("a.pdf", PDF_A), ("broken.pdf", b"%PDF-1.4 not really")], "could not be read"),
    ],
)
def test_uploads_that_break_a_limit_are_refused(files, expected):
    with pytest.raises(UploadError, match=expected):
        load_documents(files, LIMITS)


def test_a_valid_upload_keeps_numbers_pages_and_text():
    assert [(d.number, d.name, d.pages) for d in DOCUMENTS] == [(1, "a.pdf", 1), (2, "b.pdf", 1)]
    assert "measured seventeen samples" in DOCUMENTS[0].text


def test_display_names_lose_paths_and_control_characters():
    assert clean_name("C:\\tmp\\my\x07 paper.pdf", 1) == "my paper.pdf"
    assert clean_name("../../etc/passwd", 2) == "passwd"
    assert clean_name(None, 3) == "document-3.pdf"
    assert len(clean_name("x" * 500, 1)) == 120


def test_normalize_joins_hyphenated_lines_and_ignores_case_and_spacing():
    assert normalize("A meas-\n ured   Value\nhere") == "a measured value here"


# --- prompts --------------------------------------------------------------------------------


def test_a_sheet_call_carries_one_pdf_with_a_neutral_name():
    system, user = document_sheet_messages(b"one")
    assert system["content"] == SHEET_SYSTEM
    part, text = user["content"]
    assert part["type"] == "file" and text["type"] == "text"
    assert part["file"]["filename"] == "document.pdf"
    assert part["file"]["file_data"] == (
        "data:application/pdf;base64," + base64.b64encode(b"one").decode()
    )


def test_an_idea_call_reads_sheets_earlier_ideas_and_the_card_but_no_pdf():
    earlier = IdeaCard(problem_gap="g", core_claim="earlier claim", method="m", inspiration="i")
    system, user = idea_messages(SHEETS, [earlier], "A card.")
    assert system["content"] == IDEA_SYSTEM
    text = user["content"]
    assert isinstance(text, str) and "application/pdf" not in text
    assert '<document id="2">' in text and 'E1: "measured seventeen samples"' in text
    assert text.index("</documents>") < text.index("earlier claim") < text.index("«A card.»")


# --- sheets ---------------------------------------------------------------------------------


def test_a_sheet_numbers_its_evidence_and_checks_each_quote_in_the_pdf():
    sheet = parse_sheet(sheet_reply(("Measured  SEVENTEEN samples", "never written")), DOCUMENTS[0])
    assert [(e.id, e.found) for e in sheet.evidence] == [("E1", True), ("E2", False)]
    assert sheet.name == "a.pdf" and sheet.number == 1


def test_a_sheet_keeps_at_most_ten_quotes():
    sheet = parse_sheet(sheet_reply([f"quote {i}" for i in range(15)]), DOCUMENTS[0])
    assert len(sheet.evidence) == 10


@pytest.mark.parametrize(
    "content",
    [
        None,
        "not json",
        "[]",
        '{"title": "t"}',
        sheet_reply(()),
        sheet_reply(("   ",)),
        json.dumps(
            {"title": "t", "summary": " ", "methods": "", "limitations": "", "evidence": ["q"]}
        ),
    ],
)
def test_a_reply_that_breaks_the_contract_is_no_sheet(content):
    assert parse_sheet(content, DOCUMENTS[0]) is None


# --- sources --------------------------------------------------------------------------------


def test_sources_resolve_to_the_evidence_of_the_sheets():
    parsed = parse_sources(
        idea_reply(
            sources=[
                {"document": 1, "evidence": "E1"},
                {"document": 2, "evidence": "E2"},
                {"document": 9, "evidence": "E1"},
                {"document": 1, "evidence": "E99"},
                {"document": True, "evidence": "E1"},
                {"document": 1},
                "not an object",
            ]
        ),
        SHEETS,
    )
    assert [(s["name"], s["evidence"], s["found"]) for s in parsed["sources"]] == [
        ("a.pdf", "E1", True),
        ("b.pdf", "E2", False),
    ]
    assert parsed["sources"][0]["quote"] == "measured seventeen samples"
    assert parsed["rejected"] == 5


@pytest.mark.parametrize("content", [None, "not json", '{"ideas": []}', '{"ideas": [{}]}'])
def test_a_reply_without_sources_has_none(content):
    assert parse_sources(content, SHEETS) == {"sources": [], "rejected": 0}


# --- a set ----------------------------------------------------------------------------------


def test_each_pdf_goes_in_its_own_call_and_the_ideas_see_only_sheets(cfg):
    api = FakeAPI()
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert job.status == "done"
    sheets, ideas = api.of(SHEET_SYSTEM), api.of(IDEA_SYSTEM)
    assert len(sheets) == 2 and len(ideas) == 3
    sent = set()
    for request in sheets:
        files = [p for p in request["messages"][1]["content"] if p["type"] == "file"]
        assert len(files) == 1
        sent.add(files[0]["file"]["file_data"])
    assert len(sent) == 2
    assert all(isinstance(r["messages"][1]["content"], str) for r in ideas)
    assert all(data not in json.dumps(ideas) for data in sent)
    assert [s["name"] for s in job.sheets] == ["a.pdf", "b.pdf"]


def test_a_set_has_three_distinct_cards_and_each_idea_sees_the_earlier_ones(cfg):
    api = FakeAPI()
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert len({i["card"]["id"] for i in job.ideas}) == 3
    deck = load_deck(DECK_DIR)
    assert all(i["card"]["text"] == deck[i["card"]["id"]].text for i in job.ideas)
    texts = [r["messages"][1]["content"] for r in api.of(IDEA_SYSTEM)]
    assert "claim 1" not in texts[0]
    assert "claim 1" in texts[1] and "claim 2" not in texts[1]
    assert "claim 1" in texts[2] and "claim 2" in texts[2]
    assert all(r["model"] == "muse-spark-1.3" for r in api.requests)
    assert job.ideas[0]["sources"][0]["name"] == "a.pdf"
    assert job.ideas[0]["idea"]["core_claim"] == "claim 1"


def test_sheets_and_ideas_use_their_own_parameters(cfg):
    api = FakeAPI()
    run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert all(r["temperature"] == 0 for r in api.of(SHEET_SYSTEM))
    assert all(r["temperature"] == 1.0 for r in api.of(IDEA_SYSTEM))


def test_a_broken_sheet_is_retried_once(cfg):
    api = FakeAPI(sheet=lambda n: "not json" if n == 1 else None)
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert job.status == "done" and len(api.of(SHEET_SYSTEM)) == 3


def test_a_document_without_a_sheet_stops_the_set_and_refunds_it(cfg):
    api = FakeAPI(sheet=lambda n: "not json")
    service = ProductService(cfg, transport=api, wait=wait_none())
    job = run_set(service)
    assert job.status == "failed" and "could not be read into a sheet" in job.message
    assert api.of(IDEA_SYSTEM) == []
    assert service.session("s").sets_started == 0


def test_a_filtered_document_is_named_and_no_idea_is_written(cfg):
    api = FakeAPI(sheet=lambda n: filtered())
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert job.status == "failed" and "content filter refused a.pdf" in job.message
    assert api.of(IDEA_SYSTEM) == []


def test_invalid_and_filtered_ideas_are_shown_and_the_set_goes_on(cfg):
    # Calls 1 and 2: filtered twice. Calls 3 and 4: not JSON twice. Call 5: fine.
    script = {1: filtered(), 2: filtered(), 3: "nope", 4: "nope"}
    api = FakeAPI(idea=lambda n: script.get(n))
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert [i["status"] for i in job.ideas] == ["filtered", "invalid", "ok"]
    assert job.status == "done"
    assert "[Evidence Gap]" not in api.of(IDEA_SYSTEM)[-1]["messages"][1]["content"]


def test_a_failed_call_is_reported_without_details_and_refunds_the_set(cfg):
    api = FakeAPI(idea=lambda n: RuntimeError("secret detail"))
    service = ProductService(cfg, transport=api, wait=wait_none())
    job = run_set(service)
    assert job.status == "failed"
    assert "secret" not in job.message
    assert service.session("s").sets_started == 0


def test_contributor_and_control_decks_are_refused(cfg):
    bad = {**cfg, "generator": {**cfg["generator"], "model": "muse-spark-1.3-contributor"}}
    with pytest.raises(SystemExit, match="Standard"):
        ProductService(bad, transport=FakeAPI())
    bad = {**cfg, "deck": {"note": "", "cards": ["OB-001", "OB-004", "CTRL-R-001"]}}
    with pytest.raises(SystemExit, match="no controls"):
        ProductService(bad, transport=FakeAPI())
    bad = {**cfg, "deck": {"note": "", "cards": ["OB-001", "OB-004"]}}
    with pytest.raises(SystemExit, match="at least 3"):
        ProductService(bad, transport=FakeAPI())


def test_the_committed_config_uses_standard_fifty_pages_and_the_deck_of_split_b():
    cfg = yaml.safe_load(Path("configs/product.yaml").read_text("utf-8"))
    assert cfg["generator"]["model"] == "muse-spark-1.3"
    assert (cfg["limits"]["min_documents"], cfg["limits"]["max_documents"]) == (2, 5)
    assert cfg["limits"]["max_pages_per_document"] == 50
    deck = load_deck(DECK_DIR)
    # ADR-011: the rule of ADR-010, amendment 3, gives E6, marked experimental.
    assert cfg["deck"]["name"] == "E6" and cfg["deck"]["experimental"] is True
    assert sorted(cfg["deck"]["cards"]) == sorted(c for c in deck if deck[c].origin == "paper")


# --- views ----------------------------------------------------------------------------------


def test_the_default_set_is_sequential_with_cards_and_no_one_call(cfg):
    api = FakeAPI()
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert job.public()["view"] == "sequential" and job.public()["with_cards"] is True
    assert api.one_calls() == [] and job.one_call == []
    assert all("«" in r["messages"][1]["content"] for r in api.of(IDEA_SYSTEM))


def test_with_cards_off_the_set_is_still_sequential_but_has_no_card(cfg):
    api = FakeAPI()
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()), with_cards=False)
    assert job.status == "done" and job.public()["with_cards"] is False
    assert [i["card"] for i in job.ideas] == [None, None, None]
    texts = [r["messages"][1]["content"] for r in api.of(IDEA_SYSTEM)]
    assert len(texts) == 3 and not any("«" in t for t in texts)
    assert "claim 1" in texts[1] and "claim 2" in texts[2]


def test_side_by_side_adds_one_call_for_the_three_ideas_with_no_card(cfg):
    api = FakeAPI()
    service = ProductService(cfg, transport=api, wait=wait_none())
    job = run_set(service, view="side_by_side")
    assert job.status == "done"
    (one,) = api.one_calls()
    assert "«" not in one["messages"][1]["content"]
    assert "<documents>" in one["messages"][1]["content"]
    assert [i["status"] for i in job.one_call] == ["ok", "ok", "ok"]
    assert [i["card"] for i in job.one_call] == [None, None, None]
    assert job.one_call[2]["idea"]["core_claim"] == "claim one-2"
    assert job.one_call[0]["sources"][0]["name"] == "a.pdf"
    assert len(job.ideas) == 3 and all(i["card"] for i in job.ideas)
    rows = [json.loads(line) for line in Path(cfg["ledger"]).read_text("utf-8").splitlines()]
    assert [r["kind"] for r in rows].count("one_call") == 1
    assert service.session("s").sets_started == 1


def test_a_broken_one_call_fails_its_three_ideas_and_the_set_goes_on(cfg):
    api = FakeAPI(one_call='{"ideas": []}')
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()), view="side_by_side")
    assert [i["status"] for i in job.one_call] == ["invalid"] * 3
    assert job.status == "done" and len(job.ideas) == 3


def test_an_unknown_view_is_refused(cfg):
    service = ProductService(cfg, transport=FakeAPI(), wait=wait_none())
    with pytest.raises(ValueError, match="unknown view"):
        run_set(service, view="both")


# --- limits and spend -----------------------------------------------------------------------


def test_session_limit_counts_per_session(cfg):
    service = ProductService(cfg, transport=FakeAPI(), wait=wait_none())
    run_set(service, "s")
    run_set(service, "s")
    with pytest.raises(Exception, match="used its 2 sets"):
        run_set(service, "s")
    assert run_set(service, "other").status == "done"


def test_the_same_pdf_pays_for_its_sheet_only_once(cfg):
    api = FakeAPI()
    service = ProductService(cfg, transport=api, wait=wait_none())
    run_set(service, "s")
    run_set(service, "s")
    assert len(api.of(SHEET_SYSTEM)) == 2


def test_cost_is_recorded_by_kind_and_the_budget_survives_a_restart(cfg):
    api = FakeAPI(tokens=(1000, 200, 500))
    service = ProductService(cfg, transport=api, wait=wait_none())
    run_set(service)
    per_call = (800 * 1.25 + 200 * 0.15 + 500 * 4.25) / 1e6
    assert service.spent_usd == pytest.approx(5 * per_call)
    assert service.session("s").spent_usd == pytest.approx(5 * per_call)
    rows = [json.loads(line) for line in Path(cfg["ledger"]).read_text("utf-8").splitlines()]
    assert [r["kind"] for r in rows].count("sheet") == 2
    assert [r["kind"] for r in rows].count("idea") == 3

    limited = {**cfg, "limits": {**LIMITS, "budget_usd": 5 * per_call}}
    restarted = ProductService(limited, transport=api, wait=wait_none())
    assert restarted.spent_usd == pytest.approx(5 * per_call)
    with pytest.raises(Exception, match="budget"):
        run_set(restarted, "new")


def test_sheets_are_read_one_at_a_time_in_upload_order(cfg):
    class Tracking(FakeAPI):
        in_flight = 0
        most = 0

        async def create(self, **request):
            type(self).in_flight += 1
            type(self).most = max(self.most, self.in_flight)
            await asyncio.sleep(0)
            try:
                return await super().create(**request)
            finally:
                type(self).in_flight -= 1

    api = Tracking()
    run_set(ProductService(cfg, transport=api, wait=wait_none()))
    assert Tracking.most == 1
    sent = [r["messages"][1]["content"][0]["file"]["file_data"] for r in api.of(SHEET_SYSTEM)]
    assert sent == [
        "data:application/pdf;base64," + base64.b64encode(d.data).decode() for d in DOCUMENTS
    ]


def test_the_session_spending_limit_stops_between_sheets(cfg):
    limited = {**cfg, "limits": {**LIMITS, "session_usd": 0.0001}}
    api = FakeAPI()
    job = run_set(ProductService(limited, transport=api, wait=wait_none()))
    assert job.status == "limit"
    assert len(api.of(SHEET_SYSTEM)) == 1
    assert api.of(IDEA_SYSTEM) == []
    assert job.sheets == [] and job.ideas == []


def test_the_budget_stops_between_sheets_and_survives_in_the_ledger(cfg):
    limited = {**cfg, "limits": {**LIMITS, "budget_usd": 0.0001}}
    api = FakeAPI()
    service = ProductService(limited, transport=api, wait=wait_none())
    assert run_set(service).status == "limit"
    assert len(api.of(SHEET_SYSTEM)) == 1
    assert len(Path(cfg["ledger"]).read_text("utf-8").splitlines()) == 1


def test_no_pdf_reaches_the_disk(cfg):
    run_set(ProductService(cfg, transport=FakeAPI(), wait=wait_none()))
    encoded = base64.b64encode(PDF_A).decode()
    with sqlite3.connect(cfg["cache"]) as db:
        rows = db.execute("SELECT request, response FROM cache").fetchall()
    assert rows and all(encoded not in req and encoded not in resp for req, resp in rows)
    assert encoded not in Path(cfg["ledger"]).read_text("utf-8")


# --- the API --------------------------------------------------------------------------------


def wait_done(client, job_id):
    for _ in range(200):
        body = client.get(f"/api/sets/{job_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError("the set did not finish")


def upload(client, files=PAIR):
    return client.post(
        "/api/sets",
        files=[("files", (name, data, "application/pdf")) for name, data in files],
    )


def test_page_and_status(cfg):
    cfg["deck"] |= {"name": "E6", "experimental": True}
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        page = client.get("/").text
        assert "Bring your papers" in page and "One call (K=3)" in page
        status = client.get("/api/status").json()
        assert status["sets_left"] == 2 and status["deck_note"] == "Provisional."
        assert status["deck"] == "E6" and status["deck_experimental"] is True
        assert COOKIE in client.cookies


def test_the_view_and_the_cards_come_from_the_form(cfg):
    api = FakeAPI()
    with TestClient(create_app(cfg, transport=api, wait=wait_none())) as client:
        files = [("files", (name, data, "application/pdf")) for name, data in PAIR]
        started = client.post(
            "/api/sets", files=files, data={"view": "side_by_side", "cards": "off"}
        )
        body = wait_done(client, started.json()["id"])
        assert body["view"] == "side_by_side" and body["with_cards"] is False
        assert len(body["one_call"]) == 3 and len(body["ideas"]) == 3
        refused = client.post("/api/sets", files=files, data={"view": "both"})
        assert refused.status_code == 400
        assert client.get("/api/status").json()["sets_left"] == 1
    assert len(api.one_calls()) == 1


def test_upload_then_poll_until_the_set_is_done(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        started = upload(client)
        assert started.status_code == 202
        body = wait_done(client, started.json()["id"])
        assert body["status"] == "done" and len(body["ideas"]) == 3
        assert body["phase"] == "ideas" and len(body["sheets"]) == 2
        assert body["documents"][0]["name"] == "a.pdf"
        assert client.get("/api/status").json()["sets_left"] == 1


def test_a_pdf_past_the_page_limit_is_refused_with_no_call_and_no_charge(cfg):
    api = FakeAPI()
    long_paper = [("a.pdf", PDF_A), ("long.pdf", make_pdf("x", pages=4))]
    with TestClient(create_app(cfg, transport=api, wait=wait_none())) as client:
        response = upload(client, long_paper)
        assert response.status_code == 400
        assert response.json()["detail"] == "long.pdf: 4 pages; the limit is 3 per document."
        assert client.get("/api/status").json()["sets_left"] == 2
    assert api.requests == []
    assert not Path(cfg["ledger"]).exists()


def test_a_bad_upload_is_a_400_and_does_not_use_a_set(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        response = upload(client, PAIR[:1])
        assert response.status_code == 400 and "between 2 and 5" in response.json()["detail"]
        assert client.get("/api/status").json()["sets_left"] == 2


def test_the_session_limit_is_a_429(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        for _ in range(2):
            wait_done(client, upload(client).json()["id"])
        response = upload(client)
        assert response.status_code == 429 and "used its 2 sets" in response.json()["detail"]


def test_another_session_cannot_read_a_set(cfg):
    app = create_app(cfg, transport=FakeAPI(), wait=wait_none())
    with TestClient(app) as owner:
        job_id = upload(owner).json()["id"]
        wait_done(owner, job_id)
        owner.cookies.clear()
        assert owner.get(f"/api/sets/{job_id}").status_code == 404


def test_an_oversized_body_is_refused_before_parsing(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        response = client.post("/api/sets", content=b"x", headers={"content-length": str(10**9)})
        assert response.status_code == 413
