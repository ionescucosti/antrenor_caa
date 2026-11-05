"""Antrenor CAA (the study web app) served by the merged FastAPI server.

The original stdlib HTTP handler in the project root (`app.py`) is ported to a
FastAPI router; its `Application`, `Engine` and `importer` are reused as they
are, so behaviour, storage (`date/`) and the UI (`static/`) stay the same."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from rokid_backend.core.network import is_local_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as antrenor  # noqa: E402  - the root app.py (Antrenor CAA)
from engine import UserError  # noqa: E402
from importer import ImportFailure  # noqa: E402

router = APIRouter(tags=["antrenor-caa"], include_in_schema=False)
STATIC = PROJECT_ROOT / "static"
CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")


def create_application(directory: Path) -> "antrenor.Application":
    return antrenor.Application(directory, 0)


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _authorized(request: Request) -> bool:
    """Loopback clients use the app as before. Remote clients (through Cloudflare
    Tunnel or any reverse proxy, detected by the proxy headers even when the
    socket is loopback) must hold an admin session or admin bearer token."""
    if is_local_client(request):
        return True
    from rokid_backend.api.deps import SESSION_COOKIE, _bearer, _lookup_token
    session = request.app.state.signer.verify(request.cookies.get(SESSION_COOKIE))
    if session and session.get("scope") == "admin":
        return True
    token = _bearer(request)
    if token:
        principal = _lookup_token(request, token)
        return principal is not None and principal.scope == "admin"
    return False


def _guard(request: Request) -> JSONResponse | None:
    if not request.app.state.settings.caa_enabled:
        return _error("Antrenor CAA este dezactivat pe acest server.", 404)
    if not _authorized(request):
        return _error("Acces permis doar local sau cu sesiune de administrator (autentifica-te la /admin).", 403)
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        return _error("Cerere externa refuzata.", 403)
    return None


def _handle(fn):
    try:
        return fn()
    except (UserError, ImportFailure, json.JSONDecodeError, ValueError) as exc:
        return _error(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger("caa").exception("Antrenor CAA request failed: %s", type(exc).__name__)
        return _error("Operatia a esuat. Datele salvate nu au fost sterse. Verifica terminalul.", 500)


def _attachment(data: bytes, media_type: str, filename: str | None = None) -> Response:
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if filename else {}
    return Response(content=data, media_type=media_type, headers=headers)


# --- UI ----------------------------------------------------------------------

@router.get("/")
@router.get("/index.html")
def index(request: Request):
    if (denied := _guard(request)) is not None:
        if not is_local_client(request) and not request.cookies:
            return HTMLResponse("<h1>Antrenor CAA</h1><p>Acces permis doar local sau dupa autentificare la <a href='/admin'>/admin</a>.</p>", status_code=403)
        return denied
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", request.app.state.caa.token)
    return HTMLResponse(html, headers={"Content-Security-Policy": CSP})


@router.get("/app.js")
def app_js():
    return FileResponse(STATIC / "app.js", media_type="text/javascript; charset=utf-8")


@router.get("/style.css")
def style_css():
    return FileResponse(STATIC / "style.css", media_type="text/css; charset=utf-8")


@router.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@router.get("/health")
def caa_health():
    return {"app": "CAA-Trainer", "version": "1.0.0", "merged_with": "rokid-rag-backend"}


# --- GET API -----------------------------------------------------------------

@router.get("/api/state")
def state(request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    caa = request.app.state.caa
    return _handle(lambda: {**caa.engine.state(), "jobs": caa.job_state()})


@router.get("/api/stats")
def stats(request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    return _handle(lambda: request.app.state.caa.engine.statistics())


@router.get("/api/details")
def details(request: Request, kind: str = "run", id: str = ""):
    if (denied := _guard(request)) is not None:
        return denied
    return _handle(lambda: request.app.state.caa.engine.details(kind, id))


@router.get("/api/backup")
def backup(request: Request):
    if (denied := _guard(request)) is not None:
        return denied
    return _handle(lambda: _attachment(request.app.state.caa.engine.backup_zip(), "application/zip", "Backup_CAA.zip"))


@router.get("/api/export")
def export(request: Request, kind: str = "sessions"):
    if (denied := _guard(request)) is not None:
        return denied

    def run():
        if kind not in ("sessions", "runs"):
            raise UserError("Export invalid.")
        data = request.app.state.caa.engine.statistics()[kind]
        return _attachment(antrenor.csv_bytes(data, kind), "text/csv; charset=utf-8", f"CAA_{kind}.csv")
    return _handle(run)


@router.get("/api/report/{category}")
def report(request: Request, category: str):
    if (denied := _guard(request)) is not None:
        return denied

    def run():
        bank = request.app.state.caa.engine.bank_info(category) if category in ("C", "D") else None
        if not bank:
            raise UserError("Categoria nu este importata.")
        return _attachment(bank["report"].encode("utf-8"), "application/json; charset=utf-8", f"Raport_import_{category}.json")
    return _handle(run)


@router.get("/source/{category}")
def source(request: Request, category: str):
    if (denied := _guard(request)) is not None:
        return denied

    def run():
        caa = request.app.state.caa
        bank = caa.engine.bank_info(category) if category in ("C", "D") else None
        if not bank:
            raise UserError("Categoria nu este importata.")
        return Response((caa.engine.directory / "sources" / bank["source_file"]).read_bytes(), media_type="application/pdf")
    return _handle(run)


@router.get("/crop/{qid}/{index}.png")
def crop(request: Request, qid: str, index: str):
    if (denied := _guard(request)) is not None:
        return denied

    def run():
        if not re.fullmatch(r"[CD]-[0-9a-f]{10}-\d{4,6}", qid) or index not in ("0", "1", "2", "3"):
            raise UserError("Decupaj invalid.")
        return Response(request.app.state.caa.crop(qid, int(index)), media_type="image/png")
    return _handle(run)


# --- POST API ----------------------------------------------------------------

POST_LIMIT = 65536


@router.post("/api/import/{category}")
@router.post("/api/{action}")
async def post_action(request: Request, action: str = "", category: str = ""):
    if (denied := _guard(request)) is not None:
        return denied
    caa = request.app.state.caa
    is_pdf = bool(category)
    if is_pdf and category not in ("C", "D"):
        return _error("Categoria trebuie sa fie C sau D.", 400)
    limit = antrenor.MAX_PDF if is_pdf else POST_LIMIT
    raw = await request.body()
    if not raw or len(raw) > limit:
        return _error("Cerere goala sau prea mare (PDF: maximum 30 MB).", 413)
    try:
        data = {} if is_pdf else json.loads(raw)
    except json.JSONDecodeError as exc:
        return _error(str(exc), 400)
    if not isinstance(data, dict):
        return _error("Cerere JSON invalida.", 400)
    token = request.headers.get("X-CAA-Token") or data.get("token", "")
    import secrets
    if not isinstance(token, str) or not secrets.compare_digest(token, caa.token):
        return _error("Pagina este veche sau cererea nu este autorizata. Reincarca pagina.", 403)
    force = request.query_params.get("force", "0") == "1"
    engine = caa.engine

    def dispatch():
        if is_pdf:
            return JSONResponse(caa.enqueue(category, raw, force), status_code=202)
        if action == "download":
            return JSONResponse(caa.enqueue(data.get("category"), None, bool(data.get("force", False))), status_code=202)
        if action == "reset-category":
            if data.get("wipe", True):
                return engine.delete_history(data.get("category"))
            return engine.reset_category(data.get("category"))
        if action == "start":
            return engine.start(data.get("category"), data.get("topic", "all"))
        if action == "answer":
            return engine.answer(data.get("sessionId", ""), data.get("questionId", ""), data.get("selected", []))
        if action == "next":
            return engine.next_question(data.get("sessionId", ""))
        if action == "end":
            return engine.end(data.get("sessionId", ""), data.get("reason", "paused"), bool(data.get("resume", False)), data.get("foreground"))
        if action == "reset":
            return engine.reset(data.get("sessionId", ""))
        if action == "heartbeat":
            return engine.heartbeat(data.get("sessionId", ""), bool(data.get("foreground", False)))
        return _error("Operatie inexistenta.", 404)

    def run():
        try:
            return dispatch()
        except (UserError, ImportFailure, ValueError) as exc:
            return _error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("caa").exception("Antrenor CAA POST failed: %s", type(exc).__name__)
            return _error("Salvarea nu a fost confirmata. NU continua: reincarca pagina si verifica progresul. Vezi terminalul pentru detalii.", 500)
    return await run_in_threadpool(run)
