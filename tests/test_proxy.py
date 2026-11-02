"""The OpenAI-compatible proxy, against a fake provider. No API calls."""

import json
import sqlite3
from pathlib import Path

import httpx
import openai
import pytest
import yaml
from fastapi.testclient import TestClient
from tenacity import wait_none

from oblique.deck import load_deck
from oblique.deck.controls import DECK_DIR
from oblique.product.service import ProductService
from oblique.product.spend import SpendLedger
from oblique.prompts.ideation import INJECT_V1
from oblique.prompts.proxy import PREVIOUS, PROXY_SEQUENCE_V1, oblique_messages
from oblique.proxy.app import create_app, local_host
from oblique.proxy.service import ProxyService, session_of
from oblique.proxy.trial import parse_listening

OPT_IN = "muse-spark-1.3+oblique"
DECK = ["OB-001", "OB-004", "OB-005", "OB-007"]
USER = [{"role": "user", "content": "Ideas about cold chains, please."}]


class _Raw:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def filtered():
    request = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError("content management policy", response=response, body=None)


class FakeProvider:
    def __init__(self, reply=None, tokens=(1000, 200, 500)):
        self.chat = self.completions = self
        self.requests = []
        self.reply = reply or (lambda n: None)
        self.tokens = tokens

    async def create(self, **request):
        self.requests.append(request)
        n = len(self.requests)
        answer = self.reply(n)
        if isinstance(answer, Exception):
            raise answer
        prompt, cached, completion = self.tokens
        return _Raw(
            {
                "id": f"chatcmpl-{n}",
                "object": "chat.completion",
                "created": 1,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"answer {n}" if answer is None else answer,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        )


@pytest.fixture
def cfg(tmp_path):
    base = {
        "ledger": str(tmp_path / "ledger.jsonl"),
        "cache": str(tmp_path / "app.db"),
        "page": "web/product/index.html",
        "generator": {
            "provider": "meta",
            "model": "muse-spark-1.3",
            "params": {"temperature": 1.0},
            "sheet_params": {"temperature": 0},
            "timeout_seconds": 10,
            "max_attempts": 1,
        },
        "prices_per_m": {"muse-spark-1.3": {"in": 1.25, "cached": 0.15, "out": 4.25}},
        "limits": {
            "min_documents": 2,
            "max_documents": 5,
            "max_file_mb": 1,
            "max_pages_per_document": 3,
            "sets_per_session": 2,
            "session_usd": 1.0,
            "budget_usd": 10.0,
            "max_running_sets": 2,
        },
        "deck": {"note": "Provisional.", "cards": DECK},
    }
    path = tmp_path / "product.yaml"
    path.write_text(yaml.safe_dump(base), "utf-8")
    return {
        "product_config": str(path),
        "cache": str(tmp_path / "proxy.db"),
        "k": 3,
        "concurrency": 4,
        "max_tokens": 1000,
        "max_body_mb": 1,
    }


def set_limit(cfg, **limits):
    path = Path(cfg["product_config"])
    base = yaml.safe_load(path.read_text("utf-8"))
    base["limits"] |= limits
    path.write_text(yaml.safe_dump(base), "utf-8")


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def client(cfg, provider):
    with TestClient(create_app(cfg, transport=provider, wait=wait_none())) as c:
        yield c


def post(client, **body):
    return client.post("/v1/chat/completions", json=body)


def last_text(request):
    return request["messages"][-1]["content"]


# --- opt-in ---------------------------------------------------------------------------------


def test_opt_in_writes_k_answers_in_sequence_under_distinct_cards(client, provider):
    body = post(client, model=OPT_IN, messages=USER).json()
    assert len(provider.requests) == 3
    assert all(r["model"] == "muse-spark-1.3" for r in provider.requests)
    used = body["oblique"]["cards_used"]
    assert len({c["id"] for c in used}) == 3
    deck = load_deck(DECK_DIR)
    assert all(deck[c["id"]].text == c["text"] for c in used)
    for request, card in zip(provider.requests, used, strict=True):
        assert INJECT_V1.user.format(card=card["text"]).rstrip() in last_text(request)
        assert last_text(request).startswith(USER[0]["content"])
    assert "answer 1" not in last_text(provider.requests[0])
    assert "answer 1" in last_text(provider.requests[1])
    assert "answer 1" in last_text(provider.requests[2])
    assert "answer 2" in last_text(provider.requests[2])


def test_opt_in_answers_in_one_choice_with_the_usage_of_all_calls(client):
    body = post(client, model=OPT_IN, messages=USER).json()
    assert body["object"] == "chat.completion" and body["model"] == OPT_IN
    assert len(body["choices"]) == 1
    content = body["choices"][0]["message"]["content"]
    assert all(f"answer {n}" in content for n in (1, 2, 3))
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 3000,
        "completion_tokens": 1500,
        "total_tokens": 4500,
    }
    assert [a["status"] for a in body["oblique"]["answers"]] == ["ok", "ok", "ok"]
    assert body["oblique"]["deck_note"] == "Provisional."


def test_the_client_only_adds_to_a_copy_of_its_messages_and_handles_parts():
    parts = [{"role": "user", "content": [{"type": "text", "text": "Hello."}]}]
    out = oblique_messages(parts, ["first"], "A card.")
    assert parts[0]["content"] == [{"type": "text", "text": "Hello."}]
    assert out[-1]["content"][0] == {"type": "text", "text": "Hello."}
    assert "first" in out[-1]["content"][1]["text"]
    assert "«A card.»" in out[-1]["content"][1]["text"]
    assert PREVIOUS.split("\n")[0] in out[-1]["content"][1]["text"]
    assert oblique_messages(USER, [], "A card.")[-1]["content"].index("«A card.»") > 0


@pytest.mark.parametrize("param", ["tools", "response_format", "tool_choice", "logprobs"])
def test_opt_in_refuses_what_a_set_cannot_honour(client, provider, param):
    response = post(client, model=OPT_IN, messages=USER, **{param: "x"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert provider.requests == []


def test_opt_in_needs_a_last_message_from_the_user(client, provider):
    messages = [*USER, {"role": "assistant", "content": "No."}]
    assert post(client, model=OPT_IN, messages=messages).status_code == 400
    assert provider.requests == []


def test_a_filtered_answer_is_reported_and_the_set_goes_on(cfg):
    errors = {1: filtered(), 2: filtered()}  # two refusals: one identical retry
    api = FakeProvider(reply=lambda n: errors.get(n))
    with TestClient(create_app(cfg, transport=api, wait=wait_none())) as c:
        body = post(c, model=OPT_IN, messages=USER).json()
    assert [a["status"] for a in body["oblique"]["answers"]] == ["filtered", "ok", "ok"]
    assert len(body["oblique"]["cards_used"]) == 2


def test_a_set_with_every_answer_filtered_is_a_content_filter_error(cfg):
    api = FakeProvider(reply=lambda n: filtered())
    with TestClient(create_app(cfg, transport=api, wait=wait_none())) as c:
        response = post(c, model=OPT_IN, messages=USER)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "content_filter"


# --- without opt-in -------------------------------------------------------------------------


def test_without_opt_in_the_request_passes_through_unchanged(client, provider):
    tools = [{"type": "function", "function": {"name": "f"}}]
    body = post(
        client, model="muse-spark-1.3", messages=USER, temperature=0.3, tools=tools, max_tokens=50
    ).json()
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request["messages"] == USER
    assert (request["temperature"], request["tools"], request["max_tokens"]) == (0.3, tools, 50)
    assert body["choices"][0]["message"]["content"] == "answer 1"
    assert "oblique" not in body


# --- the OpenAI SDK -------------------------------------------------------------------------


def sdk(client):
    return openai.OpenAI(
        base_url="http://testserver/v1", api_key="client-key", http_client=client, max_retries=0
    )


def test_the_openai_sdk_reads_both_modes(client):
    api = sdk(client)
    plain = api.chat.completions.create(model="muse-spark-1.3", messages=USER)
    assert plain.choices[0].message.content == "answer 1"
    answer = api.chat.completions.create(model=OPT_IN, messages=USER)
    assert answer.model == OPT_IN
    assert "answer 2" in answer.choices[0].message.content
    assert len(answer.model_extra["oblique"]["cards_used"]) == 3


def test_the_openai_sdk_sees_errors_as_errors(client):
    api = sdk(client)
    with pytest.raises(openai.NotFoundError) as caught:
        api.chat.completions.create(model="gpt-4o", messages=USER)
    assert caught.value.code == "model_not_found"
    with pytest.raises(openai.BadRequestError):
        api.chat.completions.create(model=OPT_IN, messages=USER, stream=True)


# --- what is refused ------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["muse-spark-1.3-contributor", "muse-wonder-1.0", None])
def test_any_other_model_is_refused(client, provider, model):
    response = post(client, model=model, messages=USER)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert provider.requests == []


@pytest.mark.parametrize("model", [OPT_IN, "muse-spark-1.3"])
@pytest.mark.parametrize(
    "extra, param",
    [
        ({"stream": True}, "stream"),
        ({"n": 2}, "n"),
        ({"messages": []}, "messages"),
        ({"messages": "hi"}, "messages"),
        ({"messages": [{"content": "no role"}]}, "messages"),
    ],
)
def test_invalid_requests_are_refused_before_any_call(client, provider, model, extra, param):
    response = post(client, **({"model": model, "messages": USER} | extra))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert provider.requests == []


def test_bodies_that_are_not_requests_are_refused(client):
    bad = client.post("/v1/chat/completions", content=b"{not json")
    assert bad.status_code == 400 and "error" in bad.json()
    assert client.post("/v1/chat/completions", json=[1]).status_code == 400
    big = client.post("/v1/chat/completions", content=b"x", headers={"content-length": str(10**7)})
    assert big.status_code == 413


def test_a_contributor_model_or_a_control_deck_cannot_be_configured(cfg):
    path = Path(cfg["product_config"])
    base = yaml.safe_load(path.read_text("utf-8"))
    path.write_text(
        yaml.safe_dump(
            base
            | {
                "generator": {**base["generator"], "model": "muse-spark-1.3-contributor"},
                "prices_per_m": {
                    "muse-spark-1.3-contributor": base["prices_per_m"]["muse-spark-1.3"]
                },
            }
        ),
        "utf-8",
    )
    with pytest.raises(SystemExit, match="Standard"):
        ProxyService(cfg, transport=FakeProvider())
    path.write_text(yaml.safe_dump(base), "utf-8")
    bad = {**cfg, "deck": {"note": "", "cards": ["OB-001", "OB-004", "CTRL-R-001"]}}
    with pytest.raises(SystemExit, match="no controls"):
        ProxyService(bad, transport=FakeProvider())
    few = {**cfg, "deck": {"note": "", "cards": ["OB-001", "OB-004"]}}
    with pytest.raises(SystemExit, match="at least 3"):
        ProxyService(few, transport=FakeProvider())


# --- deck -----------------------------------------------------------------------------------


def test_the_deck_defaults_to_the_app_and_can_be_configured(cfg, provider):
    assert [c.id for c in ProxyService(cfg, transport=provider).deck] == DECK
    other = {**cfg, "deck": {"note": "E6.", "cards": ["E6-1", "E6-2", "E6-3"]}}
    with TestClient(create_app(other, transport=provider, wait=wait_none())) as c:
        body = post(c, model=OPT_IN, messages=USER).json()
    assert {x["id"] for x in body["oblique"]["cards_used"]} == {"E6-1", "E6-2", "E6-3"}
    assert body["oblique"]["deck_note"] == "E6."


def test_the_committed_config_shares_the_cap_and_deck_of_the_app():
    cfg = yaml.safe_load(Path("configs/proxy.yaml").read_text("utf-8"))
    assert cfg["product_config"] == "configs/product.yaml"
    assert "budget_usd" not in cfg and "ledger" not in cfg and "deck" not in cfg
    service = ProxyService(cfg | {"cache": ":memory:"}, transport=FakeProvider())
    assert service.model == "muse-spark-1.3"
    assert service.opt_in_model == OPT_IN


# --- spend ----------------------------------------------------------------------------------


def test_tokens_are_capped_for_the_credits(client, provider):
    post(client, model="muse-spark-1.3", messages=USER)
    post(client, model="muse-spark-1.3", messages=USER, max_tokens=999999, temperature=0.1)
    post(client, model="muse-spark-1.3", messages=USER, max_completion_tokens=7, temperature=0.2)
    assert [r["max_tokens"] for r in provider.requests] == [1000, 1000, 7]
    assert "max_completion_tokens" not in provider.requests[2]
    bad = post(client, model="muse-spark-1.3", messages=USER, max_tokens="many")
    assert bad.status_code == 400


def test_the_proxy_records_its_cost_in_the_ledger_of_the_app(cfg, client):
    post(client, model=OPT_IN, messages=USER)
    base = yaml.safe_load(Path(cfg["product_config"]).read_text("utf-8"))
    rows = [json.loads(x) for x in Path(base["ledger"]).read_text("utf-8").splitlines()]
    per_call = (800 * 1.25 + 200 * 0.15 + 500 * 4.25) / 1e6
    assert [r["kind"] for r in rows] == ["proxy"] * 3
    assert all(r["prompt_hash"] == PROXY_SEQUENCE_V1.hash for r in rows)
    assert sum(r["cost_usd"] for r in rows) == pytest.approx(3 * per_call)


def test_app_and_proxy_see_each_others_spend_while_both_run(cfg, client):
    base = yaml.safe_load(Path(cfg["product_config"]).read_text("utf-8"))
    app_service = ProductService(base, transport=FakeProvider(), wait=wait_none())
    assert app_service.spent_usd == 0
    post(client, model=OPT_IN, messages=USER)
    assert app_service.spent_usd > 0
    app_service.ledger.record({"key": "app", "kind": "idea", "cost_usd": 20.0})
    response = post(client, model=OPT_IN, messages=USER, temperature=0.5)
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "insufficient_quota"


def test_the_budget_survives_a_restart_and_is_a_429(cfg, provider):
    base = yaml.safe_load(Path(cfg["product_config"]).read_text("utf-8"))
    SpendLedger(base["ledger"]).record({"key": "earlier", "cost_usd": 10.0})
    with TestClient(create_app(cfg, transport=provider, wait=wait_none())) as c:
        for model in (OPT_IN, "muse-spark-1.3"):
            response = post(c, model=model, messages=USER)
            assert response.status_code == 429
    assert provider.requests == []


def test_the_session_limit_returns_what_was_written_and_stops(cfg, provider):
    set_limit(cfg, session_usd=0.0001)
    with TestClient(create_app(cfg, transport=provider, wait=wait_none())) as c:
        body = post(c, model=OPT_IN, messages=USER).json()
        assert len(provider.requests) == 1
        assert len(body["oblique"]["cards_used"]) == 1
        assert "spending limit" in body["oblique"]["stopped"]
        again = post(c, model=OPT_IN, messages=USER, temperature=0.5)
        assert again.status_code == 429
        other = c.post(
            "/v1/chat/completions",
            json={"model": "muse-spark-1.3", "messages": USER},
            headers={"authorization": "Bearer other"},
        )
        assert other.status_code == 200


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.org", "::"])
def test_the_proxy_refuses_to_listen_beyond_localhost(host):
    with pytest.raises(SystemExit, match="only on localhost"):
        local_host({"host": host})


def test_the_proxy_listens_on_localhost_by_default_and_in_the_committed_config():
    assert local_host({}) == "127.0.0.1"
    cfg = yaml.safe_load(Path("configs/proxy.yaml").read_text("utf-8"))
    assert local_host(cfg) == "127.0.0.1"


def test_the_listener_check_does_not_depend_on_the_system_language():
    netstat = """
  Proto  Dirección local  Dirección remota  Estado
  TCP    127.0.0.1:8001   0.0.0.0:0         ESCUCHANDO
  TCP    127.0.0.1:8001   127.0.0.1:50000   ESTABLISHED
  TCP    0.0.0.0:9000     0.0.0.0:0         LISTENING
"""
    assert parse_listening(netstat, 8001) == ["127.0.0.1:8001"]
    assert parse_listening(netstat, 9000) == ["0.0.0.0:9000"]
    assert parse_listening(netstat, 1234) == []


def test_sessions_come_from_a_hash_of_the_key():
    assert session_of("Bearer a") != session_of("Bearer b")
    assert "Bearer" not in session_of("Bearer a") and session_of(None) == session_of(None)


def test_the_requests_never_reach_the_disk(cfg, client):
    secret = "A private idea about cold chains."
    post(client, model=OPT_IN, messages=[{"role": "user", "content": secret}])
    with sqlite3.connect(cfg["cache"]) as db:
        rows = db.execute("SELECT request FROM cache").fetchall()
    assert rows and all(secret not in r[0] for r in rows)


def test_a_provider_failure_gives_no_details(cfg):
    request = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
    api = FakeProvider(
        reply=lambda n: openai.InternalServerError(
            "secret detail", response=httpx.Response(500, request=request), body=None
        )
    )
    with TestClient(create_app(cfg, transport=api, wait=wait_none())) as c:
        response = post(c, model="muse-spark-1.3", messages=USER)
    assert response.status_code == 502
    assert "secret" not in response.text


def test_a_rejected_request_gives_no_details(cfg):
    request = httpx.Request("POST", "https://api.meta.ai/v1/chat/completions")
    api = FakeProvider(
        reply=lambda n: openai.BadRequestError(
            "secret detail", response=httpx.Response(400, request=request), body=None
        )
    )
    with TestClient(create_app(cfg, transport=api, wait=wait_none())) as c:
        response = post(c, model="muse-spark-1.3", messages=USER)
    assert response.status_code == 400
    assert "secret" not in response.text
