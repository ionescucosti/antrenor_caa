from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import func, select

from rokid_backend.api.deps import require_admin
from rokid_backend.api.routes_documents import _read_upload
from rokid_backend.models import QADataset, QARecord
from rokid_backend.qa.service import QAError
from rokid_backend.schemas.api import QADatasetOut, QARecordOut

router = APIRouter(prefix="/api/v1/qa-datasets", tags=["qa-datasets"], dependencies=[Depends(require_admin)])


def _out(ds: QADataset) -> QADatasetOut:
    return QADatasetOut.model_validate(ds, from_attributes=True)


@router.post("", status_code=202)
async def upload_dataset(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
                         knowledge_base_id: str = Form("default"), name: str = Form(""), anr_category: str | None = Form(None)):
    service = request.app.state.qa
    data = await _read_upload(request, file)
    try:
        dataset, duplicate = service.register_upload(knowledge_base_id, file.filename or "dataset", data, name, anr_category or None)
    except QAError as exc:
        raise HTTPException(400, str(exc)) from exc
    if duplicate:
        return {"dataset": _out(dataset), "duplicate": True, "job_id": None, "message": "Set identic deja indexat."}
    job_id = service.create_job(dataset.id)
    background.add_task(service.index_dataset, dataset.id, job_id)
    return {"dataset": _out(dataset), "duplicate": False, "job_id": job_id, "message": "Set acceptat; indexarea ruleaza."}


@router.post("/sync-caa", status_code=202)
def sync_from_antrenor_caa(request: Request, background: BackgroundTasks, knowledge_base_id: str = Form("default")):
    """Reuse the ANR question PDFs already imported in Antrenor CAA (categories C
    and D) as verified Q&A datasets, without uploading them again."""
    caa = getattr(request.app.state, "caa", None)
    if caa is None:
        raise HTTPException(404, "Antrenor CAA nu este activ pe acest server.")
    service = request.app.state.qa
    results = []
    for category in ("C", "D"):
        bank = caa.engine.bank_info(category)
        if not bank:
            results.append({"category": category, "status": "missing", "message": "Categoria nu este importata in Antrenor CAA."})
            continue
        path = caa.engine.directory / "sources" / bank["source_file"]
        if not path.exists():
            results.append({"category": category, "status": "missing", "message": "PDF-ul sursa lipseste de pe disc."})
            continue
        try:
            dataset, duplicate = service.register_upload(knowledge_base_id, f"Lista intrebari clasa {category}.pdf", path.read_bytes(),
                                                         f"ANR clasa {category} (din Antrenor CAA)", category)
        except QAError as exc:
            raise HTTPException(400, str(exc)) from exc
        if duplicate:
            results.append({"category": category, "status": "unchanged", "dataset_id": dataset.id, "message": "Deja sincronizat (continut identic)."})
            continue
        job_id = service.create_job(dataset.id)
        background.add_task(service.index_dataset, dataset.id, job_id)
        results.append({"category": category, "status": "queued", "dataset_id": dataset.id, "job_id": job_id, "message": "Indexarea ruleaza."})
    return {"results": results}


@router.get("", response_model=list[QADatasetOut])
def list_datasets(request: Request, knowledge_base_id: str | None = None):
    with request.app.state.db.session() as session:
        query = select(QADataset).order_by(QADataset.created_at.desc())
        if knowledge_base_id:
            query = query.where(QADataset.kb_id == knowledge_base_id)
        return [_out(d) for d in session.execute(query).scalars()]


@router.get("/{dataset_id}", response_model=QADatasetOut)
def get_dataset(dataset_id: str, request: Request):
    try:
        return _out(request.app.state.qa.get(dataset_id))
    except QAError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{dataset_id}/records")
def list_records(dataset_id: str, request: Request, offset: int = 0, limit: int = 50, q: str | None = None):
    limit = max(1, min(limit, 200))
    with request.app.state.db.session() as session:
        base = select(QARecord).where(QARecord.dataset_id == dataset_id)
        if q:
            base = base.where(QARecord.normalized_question.contains(q.lower()))
        total = session.execute(select(func.count()).select_from(base.subquery())).scalar() or 0
        rows = session.execute(base.order_by(QARecord.created_at).offset(offset).limit(limit)).scalars()
        return {"total": total, "offset": offset, "limit": limit,
                "records": [QARecordOut.model_validate(r, from_attributes=True) for r in rows]}


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, request: Request):
    try:
        request.app.state.qa.delete(dataset_id)
    except QAError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{dataset_id}/reindex", status_code=202)
def reindex_dataset(dataset_id: str, request: Request, background: BackgroundTasks):
    service = request.app.state.qa
    try:
        job_id = service.reindex(dataset_id)
    except QAError as exc:
        raise HTTPException(404, str(exc)) from exc
    background.add_task(service.index_dataset, dataset_id, job_id)
    return {"job_id": job_id, "message": "Reindexarea ruleaza."}
