from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func, select

from rokid_backend.api.deps import SESSION_COOKIE, Principal, current_principal, require_admin
from rokid_backend.core.security import SCOPE_ADMIN, generate_token, hash_token, verify_password
from rokid_backend.models import ApiToken, Document, IngestionJob, QADataset, QARecord, new_id
from rokid_backend.schemas.api import JobOut, LoginIn, TokenIn, TokenOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/login")
def login(payload: LoginIn, request: Request, response: Response):
    settings = request.app.state.settings
    limiter = request.app.state.login_limiter
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(429, "Prea multe incercari de autentificare. Reincearca peste un minut.")
    if not settings.admin_password:
        raise HTTPException(503, "Autentificarea admin nu este configurata (ROKID_ADMIN_PASSWORD).")
    if payload.username != settings.admin_username or not verify_password(payload.password, settings.admin_password):
        log.warning("admin login failed", extra={"client": client})
        raise HTTPException(401, "Utilizator sau parola incorecte.")
    value = request.app.state.signer.sign({"scope": SCOPE_ADMIN, "user": payload.username}, settings.session_hours * 3600)
    response.set_cookie(SESSION_COOKIE, value, max_age=settings.session_hours * 3600, httponly=True,
                        samesite="strict", secure=settings.cookie_secure, path="/")
    return {"ok": True, "user": payload.username}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(principal: Principal = Depends(current_principal)):
    return {"scope": principal.scope, "name": principal.name}


@router.get("/status", dependencies=[Depends(require_admin)])
def status(request: Request):
    state = request.app.state
    with state.db.session() as session:
        docs = session.execute(select(Document.status, func.count()).group_by(Document.status)).all()
        datasets = session.execute(select(QADataset.status, func.count()).group_by(QADataset.status)).all()
        records = session.execute(select(func.count()).select_from(QARecord)).scalar() or 0
        failed_docs = [{"id": d.id, "filename": d.filename, "error": d.error} for d in
                       session.execute(select(Document).where(Document.status == "failed").limit(20)).scalars()]
        failed_sets = [{"id": d.id, "name": d.name, "error": d.error} for d in
                       session.execute(select(QADataset).where(QADataset.status == "failed").limit(20)).scalars()]
        running = session.execute(select(func.count()).select_from(IngestionJob).where(IngestionJob.status.in_(["queued", "running"]))).scalar() or 0
    return {"documents": dict(docs), "qa_datasets": dict(datasets), "qa_records": records, "jobs_running": running,
            "failed_documents": failed_docs, "failed_datasets": failed_sets,
            "llm_provider": state.llm.name, "embedding_provider": state.embedder.name,
            "general_knowledge_fallback": state.settings.allow_general_knowledge,
            "data_dir": str(state.settings.data_dir), "version": state.settings.version, "time": time.time(),
            "database_url": state.settings.effective_database_url.split("@")[-1] if not getattr(state, "database_warning", "") else "sqlite (fallback)",
            "database_warning": getattr(state, "database_warning", ""), "llm_model": state.settings.llm_model_name}


@router.get("/jobs", response_model=list[JobOut], dependencies=[Depends(require_admin)])
def jobs(request: Request, limit: int = 50):
    with request.app.state.db.session() as session:
        rows = session.execute(select(IngestionJob).order_by(IngestionJob.started_at.desc()).limit(max(1, min(limit, 500)))).scalars()
        return [JobOut.model_validate(j, from_attributes=True) for j in rows]


@router.get("/jobs/{job_id}", response_model=JobOut, dependencies=[Depends(require_admin)])
def job(job_id: str, request: Request):
    with request.app.state.db.session() as session:
        row = session.get(IngestionJob, job_id)
        if row is None:
            raise HTTPException(404, "Jobul nu exista.")
        return JobOut.model_validate(row, from_attributes=True)


@router.get("/tokens", response_model=list[TokenOut], dependencies=[Depends(require_admin)])
def list_tokens(request: Request):
    with request.app.state.db.session() as session:
        rows = session.execute(select(ApiToken).order_by(ApiToken.created_at.desc())).scalars()
        return [TokenOut.model_validate(t, from_attributes=True) for t in rows]


@router.post("/tokens", response_model=TokenOut, status_code=201, dependencies=[Depends(require_admin)])
def create_token(payload: TokenIn, request: Request):
    raw = generate_token("rk" if payload.scope == "device" else "rka")
    with request.app.state.db.session() as session:
        token = ApiToken(id=new_id("tok"), name=payload.name, token_hash=hash_token(raw), scope=payload.scope)
        session.add(token)
        session.flush()
        out = TokenOut.model_validate(token, from_attributes=True)
    out.token = raw
    return out


@router.delete("/tokens/{token_id}", status_code=204, dependencies=[Depends(require_admin)])
def revoke_token(token_id: str, request: Request):
    with request.app.state.db.session() as session:
        token = session.get(ApiToken, token_id)
        if token is None:
            raise HTTPException(404, "Tokenul nu exista.")
        token.revoked = True
