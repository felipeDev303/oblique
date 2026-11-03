"""The results page: its data comes from stored metrics, and the app links and serves it."""

import json
from pathlib import Path

import test_product
from fastapi.testclient import TestClient
from tenacity import wait_none
from test_product import FakeAPI

from oblique.export.results import build
from oblique.product.app import create_app

cfg = test_product.cfg  # the fixture of the product tests


def _ci(mean):
    return {"mean": mean, "lo": mean - 0.1, "hi": mean + 0.1, "n": 50}


def test_the_export_keeps_c1_c1s_and_every_map_point_but_c2():
    confirm = {
        "run_id": "R-B",
        "conditions": {
            c: {"cu": _ci(0.3), "q_controlled": _ci(0.4), "d": _ci(0.6), "sets_complete": 49}
            for c in ("C1", "C1s")
        },
    }
    post_b = {"n_boot": 2000, "c1s_vs_c1": {"cu": _ci(0.24), "d_embed": _ci(-0.03)}}
    point = {"q_controlled_minus_c2": _ci(0.0), "d_minus_c2": _ci(0.1), "confirmed": False}
    mapped = {
        "run_id": "R-A",
        "reference": {"delta": 0.05, "q_controlled_c2": _ci(0.5), "d_c2": _ci(0.2)},
        "ml": {"cards": ["OB-001"]},
        "conditions": [
            point | {"condition": "OB-001", "text": "t", "origin": "seed", "family": "f",
                     "band": "liminal"},
            point | {"condition": "C2", "text": None, "origin": "no_card", "family": None,
                     "band": "reference"},
            point | {"condition": "C2-T", "text": None, "origin": "no_card", "family": None,
                     "band": "inert"},
        ],
    }  # fmt: skip
    data = build(confirm, post_b, mapped)
    assert [p["id"] for p in data["map"]["cards"]] == ["OB-001", "C2-T"]
    assert data["c1_vs_c1s"]["delta"]["cu"]["mean"] == 0.24
    assert data["c1_vs_c1s"]["conditions"]["C1s"]["sets"] == 49
    assert data["map"]["c2"]["d"]["mean"] == 0.2 and data["map"]["ml"] == ["OB-001"]


def test_the_committed_data_matches_the_reported_results():
    text = Path("web/results/data.js").read_text("utf-8")
    data = json.loads(text.removeprefix("window.RESULTS = ").rstrip().removesuffix(";"))
    delta = data["c1_vs_c1s"]["delta"]
    assert (
        round(delta["cu"]["mean"], 3) == 0.240 and round(delta["q_controlled"]["mean"], 3) == 0.262
    )
    assert round(delta["d_embed"]["mean"], 3) == -0.035
    cards = data["map"]["cards"]
    assert len(cards) == 47 and any(p["id"] == "C2-T" for p in cards)
    bands = [p["band"] for p in cards if p["id"] != "C2-T"]
    assert (bands.count("liminal"), bands.count("destructive")) == (30, 11)


def test_the_app_links_and_serves_the_results_page(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        assert 'href="/results/"' in client.get("/").text
        page = client.get("/results/")
        assert page.status_code == 200 and "Oblique results" in page.text
        assert client.get("/results/data.js").text.startswith("window.RESULTS = ")
