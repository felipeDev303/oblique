"""OpenAI-compatible endpoint: a client only changes its base URL and model name.

    uv run --group product python -m oblique.proxy.app configs/proxy.yaml

Listens on localhost by default. It holds the provider's key and accepts any client key, so it must
not be exposed without an access control in front of it; the budget is the only hard stop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from oblique.proxy.service import ProxyError, ProxyService, session_of
from oblique.smoke import load_env


def create_app(cfg: dict[str, Any], *, transport: Any = None, wait: Any = None) -> FastAPI:
    service = ProxyService(cfg, transport=transport, wait=wait)
    max_body = cfg["max_body_mb"] * 1_000_000
    app = FastAPI(title="Oblique proxy", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    @app.exception_handler(ProxyError)
    async def proxy_error(_: Request, e: ProxyError) -> JSONResponse:
        return JSONResponse(e.body, status_code=e.status)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        length = request.headers.get("content-length")
        if length is None or not length.isdigit():
            raise ProxyError(411, "Content-Length required.")
        if int(length) > max_body:
            raise ProxyError(413, f"The request is larger than {cfg['max_body_mb']} MB.")
        try:
            body = await request.json()
        except ValueError as e:
            raise ProxyError(400, "The body is not valid JSON.") from e
        if not isinstance(body, dict):
            raise ProxyError(400, "The body must be a JSON object.")
        return await service.complete(body, session_of(request.headers.get("authorization")))

    return app


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.proxy.app configs/proxy.yaml")
    import uvicorn

    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    uvicorn.run(create_app(cfg), host=cfg.get("host", "127.0.0.1"), port=cfg.get("port", 8001))


if __name__ == "__main__":
    main(sys.argv[1:])
