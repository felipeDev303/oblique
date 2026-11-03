"""The Hugging Face Space: what is uploaded, its config, the ledger on storage and the cookie."""

import importlib.util
from pathlib import Path

import pytest
import test_product
import yaml
from fastapi.testclient import TestClient
from tenacity import wait_none
from test_product import FakeAPI

from oblique.product.app import check_persistent, create_app

cfg = test_product.cfg  # the fixture of the product tests

SPEC = importlib.util.spec_from_file_location("space_build", "deploy/space/build.py")
space_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(space_build)


def test_the_space_folder_holds_the_app_and_no_secret_or_data(tmp_path):
    copied = {p.as_posix() for p in space_build.build(tmp_path / "space")}
    for needed in (
        "Dockerfile",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "configs/space.yaml",
        "src/oblique/product/app.py",
        "deck/cards/E6-1.yaml",
        "web/product/index.html",
    ):
        assert needed in copied
    assert not any(p.endswith((".env", ".db", ".jsonl")) for p in copied)
    assert not any(p.startswith(("data/", "runs/", ".cache/", "configs/R-")) for p in copied)
    readme = (tmp_path / "space" / "README.md").read_text("utf-8")
    assert "sdk: docker" in readme and "app_port: 7860" in readme
    with pytest.raises(SystemExit, match="not empty"):
        space_build.build(tmp_path / "space")


def test_the_space_config_caps_spend_and_keeps_the_ledger_on_storage():
    space = yaml.safe_load(Path("configs/space.yaml").read_text("utf-8"))
    product = yaml.safe_load(Path("configs/product.yaml").read_text("utf-8"))
    assert space["limits"]["budget_usd"] <= 10
    assert space["ledger"].startswith(space["persistent_dir"] + "/")
    assert space["cache"].startswith(space["persistent_dir"] + "/")
    assert space["generator"]["model"] == "muse-spark-1.3"
    assert space["cookie_cross_site"] is True and space["port"] == 7860
    assert space["deck"] == product["deck"]
    assert space["generator"] == product["generator"]


def test_the_app_refuses_to_start_without_persistent_storage(tmp_path):
    outside = str(tmp_path / "l.jsonl")
    with pytest.raises(SystemExit, match="restart from zero"):
        check_persistent({"persistent_dir": str(tmp_path / "missing"), "ledger": outside})
    (tmp_path / "data").mkdir()
    with pytest.raises(SystemExit, match="must be inside"):
        check_persistent({"persistent_dir": str(tmp_path / "data"), "ledger": outside})
    check_persistent({"persistent_dir": str(tmp_path / "data"), "ledger": str(tmp_path / "data/l")})
    check_persistent({"ledger": "anywhere"})


def test_the_cookie_works_inside_the_space_iframe(cfg):
    cfg["cookie_cross_site"] = True
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        header = client.get("/api/status").headers["set-cookie"].lower()
    assert "samesite=none" in header and "secure" in header and "partitioned" in header


def test_locally_the_cookie_stays_lax(cfg):
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        header = client.get("/api/status").headers["set-cookie"].lower()
    assert "samesite=lax" in header and "partitioned" not in header


def test_the_results_page_is_served_when_it_exists(cfg, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "index.html").write_text("<title>Results</title>", "utf-8")
    cfg["results_dir"] = str(results)
    with TestClient(create_app(cfg, transport=FakeAPI(), wait=wait_none())) as client:
        assert "Results" in client.get("/results/").text
