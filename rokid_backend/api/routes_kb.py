from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from rokid_backend.api.deps import require_admin
from rokid_backend.repositories.knowledge_bases import KnowledgeBaseError
from rokid_backend.schemas.api import KnowledgeBaseIn, KnowledgeBaseOut

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["knowledge-bases"], dependencies=[Depends(require_admin)])


@router.get("", response_model=list[KnowledgeBaseOut])
def list_knowledge_bases(request: Request):
    return request.app.state.knowledge_bases.list()


@router.post("", response_model=KnowledgeBaseOut, status_code=201)
def create_knowledge_base(payload: KnowledgeBaseIn, request: Request):
    try:
        return request.app.state.knowledge_bases.create(payload.id, payload.name, payload.description)
    except KnowledgeBaseError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: str, request: Request):
    try:
        request.app.state.knowledge_bases.delete(kb_id)
    except KnowledgeBaseError as exc:
        raise HTTPException(400 if "implicita" in str(exc) else 404, str(exc)) from exc
