"""What the page shows: P labels, short titles, named documents and completed passages. The model
keeps reading E labels and the ideas as it wrote them."""

import json

import pytest
import test_product
from fastapi.testclient import TestClient
from tenacity import wait_none
from test_product import DOCUMENTS, IDEA_SYSTEM, SHEETS, FakeAPI, idea_reply, run_set

from oblique.product.app import create_app
from oblique.product.documents import (
    complete_passage,
    name_documents,
    parse_sheet,
    short_title,
)
from oblique.product.examples import load_example
from oblique.product.service import ProductService, parse_sources
from oblique.prompts.product import idea_messages

cfg = test_product.cfg  # the fixture of the product tests
TEXT = "We ran it twice. In summary, the gains vanish under adversarial probes. Then we stopped."


@pytest.mark.parametrize(
    "quote, text, expected",
    [
        ("the gains vanish", TEXT, "the gains vanish under adversarial probes."),
        ("The Gains vanish", TEXT, "the gains vanish under adversarial probes."),
        ("we ran it twice.", TEXT, "we ran it twice."),
        ("not in the text", TEXT, "not in the text…"),
        ("as shown by", "as shown by Chang et al. in Fig. 3 the gap is real. End.", None),
        ("a long", "a long " + "word " * 60 + "end. Next.", "a long…"),
        ("cut here,", "", "cut here…"),
    ],
)
def test_a_passage_runs_to_its_sentence_end_or_ends_in_an_ellipsis(quote, text, expected):
    if expected is None:  # abbreviations are not sentence ends
        expected = "as shown by Chang et al. in Fig. 3 the gap is real."
    assert complete_passage(quote, text) == expected


def test_uploaded_documents_keep_their_case_to_complete_passages():
    # The test PDF's only line has no final stop: the passage keeps the PDF's case and ends in "…".
    sheet = parse_sheet(
        json.dumps(
            {
                "title": "A title",
                "summary": "s",
                "methods": "m",
                "limitations": "l",
                "evidence": ["Alpha measured seventeen"],
            }
        ),
        DOCUMENTS[0],
    )
    (evidence,) = sheet.evidence
    assert evidence.found and evidence.passage == "Alpha measured seventeen…"
    assert DOCUMENTS[0].cased == "Alpha measured seventeen samples in the cold room"
    assert DOCUMENTS[0].text == DOCUMENTS[0].cased.lower()


@pytest.mark.parametrize(
    "title, short",
    [
        ("Otter: A Multi-Modal Model with In-Context Instruction Tuning", "Otter"),
        ("A Survey on Evaluation of Large Language Models", "A Survey on Evaluation of Large…"),
        ("Short title", "Short title"),
        ("A very long title before the colon, longer than the limit: subtitle", None),
    ],
)
def test_short_titles(title, short):
    if short is None:
        assert short_title(title).endswith("…") and len(short_title(title)) <= 37
    else:
        assert short_title(title) == short


def test_documents_are_named_by_short_title_only_when_they_exist():
    shorts = {1: "Survey", 2: "Otter"}
    text = "From Document 1, documents 1 and 2, Doc. 2, doc #1, Document 3 and Document 2023."
    assert name_documents(text, shorts) == (
        "From Survey, Survey and Otter, Otter, Survey, Document 3 and Document 2023."
    )


def test_sources_and_sheets_show_p_labels_while_the_model_reads_e_labels():
    parsed = parse_sources(idea_reply(sources=[{"document": 2, "evidence": "E1"}]), SHEETS)
    (source,) = parsed["sources"]
    assert (source["label"], source["evidence"], source["short"]) == ("P1", "E1", "A title")
    assert source["passage"]
    public = SHEETS[0].public()
    assert [e["label"] for e in public["evidence"]] == ["P1", "P2"]
    text = idea_messages(SHEETS)[1]["content"]
    assert 'E1: "' in text and "P1" not in text


def test_an_idea_is_shown_with_named_documents_but_the_next_call_reads_it_as_written(cfg):
    def reply(n):
        idea = json.loads(idea_reply(str(n)))
        idea["ideas"][0]["inspiration"] = "Combines Document 1 with Document 2."
        return json.dumps(idea)

    api = FakeAPI(idea=reply)
    job = run_set(ProductService(cfg, transport=api, wait=wait_none()))
    first = job.ideas[0]
    assert first["display"]["inspiration"] == "Combines A title with A title."
    assert first["idea"]["inspiration"] == "Combines Document 1 with Document 2."
    second = api.of(IDEA_SYSTEM)[1]["messages"][1]["content"]
    assert "Combines Document 1 with Document 2." in second


def test_the_example_shows_labels_short_titles_and_whole_passages():
    example = load_example("examples/papers/example.json")
    for sheet in example.sheets:
        for e in sheet.public()["evidence"]:
            assert e["label"].startswith("P")
            assert e["passage"].rstrip('"”’)] ').endswith((".", "!", "?", "…"))
    shorts = {s.short for s in example.sheets}
    for item in example.one_call:
        assert item["display"]
        assert all(s["short"] in shorts and s["label"].startswith("P") for s in item["sources"])


def test_the_page_labels_the_claim_as_a_bet_and_folds_the_sources(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        page = client.get("/").text
    assert '"Claim (bet)"' in page and "Sources (${item.sources.length})" in page
    assert "SOURCES_OPEN = 2" in page and "${s.short} · ${s.label}" in page
