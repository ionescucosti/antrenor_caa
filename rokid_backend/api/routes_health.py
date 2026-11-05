from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health(request: Request):
    state = request.app.state
    db_ok = True
    try:
        with state.db.session() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    chroma_ok = state.store.heartbeat()
    warning = getattr(state, "database_warning", "")
    status = "ok" if db_ok and chroma_ok and not warning else "degraded"
    return {"status": status, "version": state.settings.version, "warnings": [warning] if warning else [], "components": {
        "database": ("sqlite-fallback" if warning else "ok") if db_ok else "error", "vector_store": "ok" if chroma_ok else "error",
        "antrenor_caa": "ok" if getattr(state, "caa", None) is not None else "disabled",
        "llm_provider": state.llm.name, "embedding_provider": state.embedder.name,
        "general_knowledge_fallback": state.settings.allow_general_knowledge}}
