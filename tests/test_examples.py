"""The example papers: what is packaged, and a set over them pays only the sequential ideas."""

import json
from pathlib import Path

import pytest
import test_product
from fastapi.testclient import TestClient
from tenacity import wait_none
from test_product import IDEA_SYSTEM, SHEET_SYSTEM, FakeAPI, run_set, wait_done

from oblique.product.app import create_app
from oblique.product.examples import build, current_prompts, load_example
from oblique.product.service import ProductService

cfg = test_product.cfg  # the fixture of the product tests
EXAMPLE = "examples/papers/example.json"


@pytest.fixture
def with_example(cfg):
    return cfg | {"example": EXAMPLE}


def test_the_packaged_example_is_two_cc_by_papers_with_their_sheets_and_one_call():
    data = json.loads(Path(EXAMPLE).read_text("utf-8"))
    assert data["prompts"] == current_prompts()
    assert [(p["arxiv_id"], p["version"]) for p in data["papers"]] == [
        ("2307.03109", "v9"),
        ("2305.03726", "v2"),
    ]
    assert all(p["license"] == "CC BY 4.0" and p["authors"] for p in data["papers"])
    assert all(
        p["url"] == f"https://arxiv.org/abs/{p['arxiv_id']}{p['version']}" for p in data["papers"]
    )
    example = load_example(EXAMPLE)
    assert [s.public() for s in example.sheets] == data["sheets"]
    assert [i["status"] for i in example.one_call] == ["ok", "ok", "ok"]
    assert all(i["card"] is None for i in example.one_call)
    assert not list(Path("examples/papers").glob("*.pdf"))


def test_an_example_built_with_other_prompts_or_licenses_is_refused(tmp_path):
    data = json.loads(Path(EXAMPLE).read_text("utf-8"))
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(data | {"prompts": {"sheet": "x", "one_call": "y"}}), "utf-8")
    with pytest.raises(SystemExit, match="other prompts"):
        load_example(stale)
    closed = tmp_path / "closed.json"
    papers = [p | {"license": "arXiv non-exclusive"} for p in data["papers"]]
    closed.write_text(json.dumps(data | {"papers": papers}), "utf-8")
    with pytest.raises(SystemExit, match="CC BY"):
        load_example(closed)


def test_the_example_file_comes_only_from_a_finished_side_by_side_set():
    with pytest.raises(SystemExit, match="side-by-side"):
        build({"view": "sequential", "status": "done"}, {})


def test_an_example_set_pays_only_the_sequential_ideas(with_example):
    api = FakeAPI()
    service = ProductService(with_example, transport=api, wait=wait_none())
    job = run_set(service, documents=[], view="side_by_side", example=True)
    assert job.status == "done" and job.public()["example"] is True
    assert api.of(SHEET_SYSTEM) == [] and api.one_calls() == []
    assert len(api.of(IDEA_SYSTEM)) == 3
    example = load_example(EXAMPLE)
    assert job.sheets == [s.public() for s in example.sheets]
    assert job.one_call == example.one_call
    text = api.of(IDEA_SYSTEM)[0]["messages"][1]["content"]
    assert (
        "A Survey on Evaluation of Large Language Models" in text and "application/pdf" not in text
    )
    rows = [
        json.loads(line) for line in Path(with_example["ledger"]).read_text("utf-8").splitlines()
    ]
    assert [r["kind"] for r in rows] == ["idea"] * 3
    assert service.session("s").sets_started == 1


def test_without_example_papers_an_example_set_is_refused(cfg):
    service = ProductService(cfg, transport=FakeAPI(), wait=wait_none())
    with pytest.raises(ValueError, match="no example papers"):
        run_set(service, documents=[], example=True)


def test_the_page_offers_the_example_with_its_ids_and_licenses(with_example):
    api = FakeAPI()
    with TestClient(create_app(with_example, transport=api, wait=wait_none())) as client:
        assert "Try with 2 example papers" in client.get("/").text
        example = client.get("/api/status").json()["example"]
        assert [p["arxiv_id"] for p in example["papers"]] == ["2307.03109", "2305.03726"]
        assert all(p["license"] == "CC BY 4.0" for p in example["papers"])
        assert "generated in advance" in example["precomputed"]
        started = client.post("/api/examples", data={"view": "side_by_side", "cards": "on"})
        assert started.status_code == 202
        body = wait_done(client, started.json()["id"])
        assert body["status"] == "done" and len(body["one_call"]) == 3 and len(body["ideas"]) == 3
        assert client.get("/api/status").json()["sets_left"] == 1
    assert api.of(SHEET_SYSTEM) == []


def test_without_example_papers_the_endpoint_is_a_404(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        assert client.get("/api/status").json()["example"] is None
        assert client.post("/api/examples", data={"view": "sequential"}).status_code == 404
