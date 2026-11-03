"""The page and its small API.

    uv run --group product python -m oblique.product.app configs/product.yaml

Listens on localhost by default. There are no accounts: before exposing it, keep in mind that the
session limit is a cookie, and only the budget in the config is a hard stop.

Exposed (`configs/space.yaml`, Hugging Face Spaces), two settings matter:
- `persistent_dir`: the ledger must live there, or the budget would restart from zero with every
  restart of the container. The app refuses to start if that directory does not exist.
- `cookie_cross_site`: the Space page shows the app inside an iframe of another site, where a
  `SameSite=Lax` cookie is not sent, so every request would be a new session. The cookie then goes
  out as `SameSite=None; Secure; Partitioned`.

The results page (`web/results/`) is served at `/results/`, when the directory exists.
"""

from __future__ import annotations

import asyncio
import secrets
import sys
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from oblique.product.documents import UploadError, load_documents
from oblique.product.service import VIEWS, Busy, LimitReached, ProductService
from oblique.smoke import load_env

COOKIE = "oblique_session"
OVERHEAD_BYTES = 1_000_000  # multipart boundaries and field headers


def create_app(cfg: dict[str, Any], *, transport: Any = None, wait: Any = None) -> FastAPI:
    service = ProductService(cfg, transport=transport, wait=wait)
    limits = cfg["limits"]
    max_body = limits["max_documents"] * limits["max_file_mb"] * 1_000_000 + OVERHEAD_BYTES
    app = FastAPI(title="Oblique", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service
    cross_site = bool(cfg.get("cookie_cross_site"))
    results = Path(cfg.get("results_dir", "web/results"))
    if results.is_dir():
        app.mount("/results", StaticFiles(directory=results, html=True), name="results")

    @app.middleware("http")
    async def session_and_body_limit(request: Request, call_next):
        # Refused before the multipart parser spools the body to disk.
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None or not length.isdigit():
                return JSONResponse({"detail": "Content-Length required."}, status_code=411)
            if int(length) > max_body:
                return JSONResponse({"detail": "The upload is too large."}, status_code=413)
        session_id = request.cookies.get(COOKIE)
        is_new = session_id is None
        request.state.session = session_id or secrets.token_urlsafe(24)
        response: Response = await call_next(request)
        if is_new and cross_site:
            # Written by hand: Starlette sets `Partitioned` only on Python 3.14 and later. The
            # value is URL-safe base64, so it needs no quoting.
            response.headers.append(
                "set-cookie",
                f"{COOKIE}={request.state.session}; Path=/; HttpOnly; Secure; SameSite=None; "
                "Partitioned",
            )
        elif is_new:
            response.set_cookie(COOKIE, request.state.session, httponly=True, samesite="lax")
        return response

    @app.get("/")
    async def page() -> FileResponse:
        return FileResponse(Path(cfg["page"]), media_type="text/html")

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        return service.status(request.state.session)

    @app.post("/api/sets", status_code=202)
    async def create_set(
        request: Request,
        files: list[UploadFile],
        view: str = Form("sequential"),
        cards: str = Form("on"),
    ) -> dict[str, Any]:
        session = request.state.session
        if view not in VIEWS or cards not in ("on", "off"):
            raise HTTPException(status_code=400, detail="Unknown view or card setting.")
        try:
            # Limits first, so a session that has used its sets does not make the server read PDFs.
            service.check_can_start(service.session(session))
            uploads = [(f.filename, await f.read()) for f in files]
            documents = await asyncio.to_thread(load_documents, uploads, limits)
            job = service.start(session, documents, view=view, with_cards=cards == "on")
            return job.public()
        except UploadError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except (LimitReached, Busy) as e:
            raise HTTPException(status_code=429, detail=str(e)) from e

    @app.get("/api/sets/{job_id}")
    async def get_set(job_id: str, request: Request) -> dict[str, Any]:
        job = service.jobs.get(job_id)
        if job is None or job.session != request.state.session:
            raise HTTPException(status_code=404, detail="Not found.")
        return job.public()

    return app


def check_persistent(cfg: dict[str, Any]) -> None:
    """With `persistent_dir`, the ledger must be inside it and the directory must exist."""
    if "persistent_dir" not in cfg:
        return
    base = Path(cfg["persistent_dir"])
    if not base.is_dir():
        raise SystemExit(
            f"{base} does not exist: without persistent storage the budget would restart from zero "
            "on every restart. Enable it, or remove persistent_dir from the config to accept that."
        )
    if not Path(cfg["ledger"]).resolve().is_relative_to(base.resolve()):
        raise SystemExit(f"the ledger must be inside {base}")


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m oblique.product.app configs/product.yaml")
    import uvicorn

    load_env()
    cfg = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    check_persistent(cfg)
    uvicorn.run(create_app(cfg), host=cfg.get("host", "127.0.0.1"), port=cfg.get("port", 8000))


if __name__ == "__main__":
    main(sys.argv[1:])
