from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select

from rokid_backend.api.deps import require_admin
from rokid_backend.ingestion.service import IngestionError
from rokid_backend.models import Document
from rokid_backend.schemas.api import DocumentOut

router = APIRouter(prefix="/api/v1/documents", tags=["documents"], dependencies=[Depends(require_admin)])


async def _read_upload(request: Request, upload: UploadFile) -> bytes:
    limit = request.app.state.settings.max_upload_bytes
    data = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, f"Fisierul depaseste limita de {request.app.state.settings.max_upload_mb} MB.")
    return bytes(data)


def _out(doc: Document) -> DocumentOut:
    return DocumentOut.model_validate(doc, from_attributes=True)


@router.post("", status_code=202)
async def upload_document(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
                          knowledge_base_id: str = Form("default"), replace_document_id: str | None = Form(None)):
    service = request.app.state.documents
    data = await _read_upload(request, file)
    try:
        doc, duplicate = service.register_upload(knowledge_base_id, file.filename or "upload", data,
                                                 file.content_type or "", replace_document_id)
    except IngestionError as exc:
        raise HTTPException(400, str(exc)) from exc
    if duplicate:
        return {"document": _out(doc), "duplicate": True, "job_id": None,
                "message": "Continut identic deja indexat; nu a fost reindexat."}
    job_id = service.create_job("document", doc.id)
    background.add_task(service.index_document, doc.id, job_id)
    return {"document": _out(doc), "duplicate": False, "job_id": job_id, "message": "Document acceptat; indexarea ruleaza."}


@router.get("", response_model=list[DocumentOut])
def list_documents(request: Request, knowledge_base_id: str | None = None):
    with request.app.state.db.session() as session:
        query = select(Document).order_by(Document.created_at.desc())
        if knowledge_base_id:
            query = query.where(Document.kb_id == knowledge_base_id)
        return [_out(d) for d in session.execute(query).scalars()]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, request: Request):
    try:
        return _out(request.app.state.documents.get(document_id))
    except IngestionError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, request: Request):
    try:
        request.app.state.documents.delete(document_id)
    except IngestionError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{document_id}/reindex", status_code=202)
def reindex_document(document_id: str, request: Request, background: BackgroundTasks):
    service = request.app.state.documents
    try:
        job_id = service.reindex(document_id)
    except IngestionError as exc:
        raise HTTPException(404, str(exc)) from exc
    background.add_task(service.index_document, document_id, job_id)
    return {"job_id": job_id, "message": "Reindexarea ruleaza."}
