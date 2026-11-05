"""Application factory. `uvicorn rokid_backend.main:app` serves the production app."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from rokid_backend.api import routes_admin, routes_documents, routes_health, routes_kb, routes_qa, routes_questions
from rokid_backend.caa import CSP as CAA_CSP, create_application as create_caa_application
from rokid_backend.core.idempotency import IdempotencyCache
from rokid_backend.core.network import client_ip
from rokid_backend.caa import router as caa_router
from rokid_backend.core.config import Settings, load_settings
from rokid_backend.core.logging import configure_logging
from rokid_backend.core.security import RateLimiter, SessionSigner, hash_token, load_or_create_secret
from rokid_backend.ingestion.service import DocumentService
from rokid_backend.llm.factory import build_llm_provider
from rokid_backend.models import ApiToken, new_id
from rokid_backend.qa.service import QAService
from rokid_backend.rag.embeddings import build_embedding_provider
from rokid_backend.rag.pipeline import AnswerService
from rokid_backend.rag.vectorstore import VectorStore
from rokid_backend.repositories.db import Database
from rokid_backend.repositories.knowledge_bases import KnowledgeBaseService

log = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


def _seed_env_tokens(db: Database, raw_tokens: str) -> None:
    """Device tokens given through the environment are stored hashed, like the others."""
    for index, raw in enumerate(t.strip() for t in raw_tokens.split(",") if t.strip()):
        digest = hash_token(raw)
        with db.session() as session:
            if session.execute(select(ApiToken).where(ApiToken.token_hash == digest)).scalars().first() is None:
                session.add(ApiToken(id=new_id("tok"), name=f"env-device-{index + 1}", token_hash=digest, scope="device"))


def _open_database(settings: Settings) -> tuple[Database, str]:
    """Use ROKID_DATABASE_URL / DATABASE_URL when it works; otherwise fall back
    to the local SQLite file and report it, so the server still starts."""
    url = settings.effective_database_url
    if not url.startswith("sqlite"):
        try:
            db = Database(url)
            with db.session() as session:
                session.execute(text("SELECT 1"))
            return db, ""
        except Exception as exc:  # noqa: BLE001 - any driver/connection error
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            log.error("database unavailable, falling back to SQLite: %s", reason)
            return Database(f"sqlite:///{settings.sqlite_path}"), f"Conexiunea la {url.split('@')[-1]} a esuat ({reason}); se foloseste SQLite local."
    return Database(url), ""


def build_state(app: FastAPI, settings: Settings, llm=None, embedder=None) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings
    app.state.db, app.state.database_warning = _open_database(settings)
    app.state.store = VectorStore(settings.effective_chroma_dir)
    app.state.embedder = embedder or build_embedding_provider(settings)
    app.state.llm = llm or build_llm_provider(settings)
    secret = settings.session_secret or load_or_create_secret(settings.data_dir / ".session_secret")
    app.state.signer = SessionSigner(secret)
    app.state.limiter = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
    app.state.answer_cache = IdempotencyCache(settings.idempotency_ttl_seconds)
    app.state.login_limiter = RateLimiter(10, 5)
    app.state.knowledge_bases = KnowledgeBaseService(app.state.db, app.state.store)
    app.state.knowledge_bases.ensure_default()
    app.state.documents = DocumentService(app.state.db, app.state.store, app.state.embedder, settings)
    app.state.qa = QAService(app.state.db, app.state.store, app.state.embedder, settings)
    app.state.answers = AnswerService(app.state.qa, app.state.store, app.state.embedder, app.state.llm, settings,
                                      app.state.knowledge_bases.exists)
    _seed_env_tokens(app.state.db, settings.device_tokens)
    if not settings.admin_password and not settings.admin_api_token:
        log.warning("No admin credentials configured: set ROKID_ADMIN_PASSWORD and/or ROKID_ADMIN_API_TOKEN.")
    app.state.caa = None
    if settings.caa_enabled:
        app.state.caa = create_caa_application(settings.caa_data_dir)
        if settings.caa_auto_download:
            for category in ("C", "D"):
                if not app.state.caa.engine.bank_info(category):
                    app.state.caa.enqueue(category)


def create_app(settings: Settings | None = None, llm=None, embedder=None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        build_state(app, settings, llm, embedder)
        log.info("started", extra={"scope": settings.environment})
        try:
            yield
        finally:
            if getattr(app.state, "caa", None) is not None:
                app.state.caa.engine.close()
            app.state.db.close()

    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan,
                  docs_url="/api/docs" if settings.environment != "production" else None,
                  openapi_url="/api/openapi.json" if settings.environment != "production" else None, redoc_url=None)

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def guard(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:16]
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_body_mb * 1024 * 1024:
            return JSONResponse({"detail": f"Corpul cererii depaseste {settings.max_body_mb} MB."}, status_code=413)
        if request.url.path.startswith("/api/v1/") and request.url.path != "/api/v1/health":
            auth = request.headers.get("Authorization", "")
            key = hash_token(auth)[:16] if auth else (client_ip(request) or "anon")
            if not app.state.limiter.allow(key):
                return JSONResponse({"detail": "Prea multe cereri. Incearca din nou in cateva secunde."}, status_code=429,
                                    headers={"Retry-After": "5"})
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        if settings.cookie_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.url.path.startswith("/admin"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
        elif request.url.path in ("/", "/index.html"):
            response.headers["Content-Security-Policy"] = CAA_CSP
        log.info("request", extra={"request_id": request_id, "method": request.method, "path": request.url.path,
                                   "status": response.status_code, "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
        return response

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse({"detail": "Eroare interna."}, status_code=500)

    for module in (routes_health, routes_questions, routes_documents, routes_qa, routes_kb, routes_admin):
        app.include_router(module.router)
    app.include_router(caa_router)  # Antrenor CAA UI at /, its API at /api/* (not /api/v1)

    app.mount("/admin/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/admin", include_in_schema=False)
    def admin_page():
        return FileResponse(STATIC_DIR / "admin.html")

    return app


app = create_app()
