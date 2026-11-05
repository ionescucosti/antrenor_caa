"""The one question endpoint used by the Rokid glasses and by manual tests.

Two request shapes on the same URL:
  * application/json  - the Rokid client contract (structured question, optional
    inline base64 image). Reply: the full AnswerResponse, request_id echoed.
  * multipart/form-data or x-www-form-urlencoded - `text` and/or `image` for
    curl, Swagger and the admin page. Reply: compact {label, text} / {error},
    or the full AnswerResponse with ?full=1.
Order: the text is tried first; the image is transcribed only when the text
gave no answer. Replies are idempotent per request_id (client retries)."""
from __future__ import annotations

import base64
import binascii
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from rokid_backend.api.deps import Principal, require_device
from rokid_backend.rag.transcribe import MAX_IMAGE_BYTES, TranscriptionError, transcribe_image
from rokid_backend.schemas.api import AnswerRequest, AnswerResponse, CompactAnswer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/questions", tags=["questions"])

FULL_PARAM = Query(False, description="formular: true = raspunsul detaliat (status, surse, timpi); implicit doar litera si textul. "
                                      "Cererile JSON primesc intotdeauna raspunsul complet.")


def compact(response: AnswerResponse) -> dict:
    """{"label": "A", "text": "..."} / {"text": "..."} / {"error": "..."} - nothing else."""
    if response.status == "answered" and response.answer is not None:
        labels = ",".join(response.answer.option_labels) or None
        return CompactAnswer(label=labels, text=response.answer.text or None).model_dump(exclude_none=True)
    return CompactAnswer(error=response.message or response.status).model_dump(exclude_none=True)


def _text_request(rid: str, kb: str, text: str) -> AnswerRequest:
    return AnswerRequest(request_id=rid, knowledge_base_id=kb, question=text, question_type="open", options=[], raw_ocr_text=text)


def _run_pipeline(payload: AnswerRequest, request: Request, principal: Principal) -> AnswerResponse:
    service = request.app.state.answers
    try:
        response = service.answer(payload)
    except Exception:  # noqa: BLE001 - the contract promises a structured error
        log.exception("answer pipeline crashed", extra={"request_id": payload.request_id})
        response = AnswerResponse(request_id=payload.request_id, status="error", message="Eroare interna la generarea raspunsului.")
    log.info("question answered", extra={"request_id": payload.request_id, "status": response.status,
                                         "duration_ms": response.processing_time_ms, "scope": principal.scope})
    return response


def _compute(request: Request, principal: Principal, payload: AnswerRequest | None, image: bytes | None, rid: str, kb: str) -> AnswerResponse:
    """Text first, image as fallback. Runs in a worker thread."""
    started = time.perf_counter()
    response: AnswerResponse | None = None
    if payload is not None and payload.has_text():
        response = _run_pipeline(payload, request, principal)
    if image is not None and (response is None or response.status != "answered"):
        state = request.app.state
        transcribe_started = time.perf_counter()
        try:
            transcribed, method = transcribe_image(image, state.llm, state.settings)
        except TranscriptionError as exc:
            if response is None:
                response = AnswerResponse(request_id=rid, status="invalid_question", message=str(exc))
            else:
                response.message = f"{response.message} Poza: {exc}".strip()
            log.info("image transcription failed", extra={"request_id": rid, "scope": principal.scope})
        else:
            transcribe_ms = round((time.perf_counter() - transcribe_started) * 1000, 1)
            response = _run_pipeline(_text_request(rid, kb, transcribed), request, principal)
            response.transcribed_text = transcribed
            response.timings_ms = {"transcription": transcribe_ms, **response.timings_ms}
            log.info("image answered", extra={"request_id": rid, "status": response.status, "method": method, "scope": principal.scope})
    if response is None:
        response = AnswerResponse(request_id=rid, status="invalid_question", message="Cererea nu contine nici text, nici imagine.")
    response.request_id = rid
    response.processing_time_ms = int((time.perf_counter() - started) * 1000)
    return response


async def _answer_once(request: Request, principal: Principal, rid: str, kb: str,
                       payload: AnswerRequest | None, image: bytes | None) -> AnswerResponse:
    """Idempotent per (token, request_id): retries get the stored reply; a retry
    that overlaps the first attempt waits for it instead of recomputing."""
    cache = request.app.state.answer_cache
    key = f"{principal.name}:{rid}"
    cached = cache.get(key)
    if cached is not None:
        log.info("idempotent replay", extra={"request_id": rid, "scope": principal.scope})
        return cached
    lock = cache.lock(key)
    async with lock:
        cached = cache.get(key)
        if cached is not None:
            return cached
        response = await run_in_threadpool(_compute, request, principal, payload, image, rid, kb)
        cache.put(key, response)
    cache.release(key)
    return response


async def _read_upload(upload: UploadFile) -> bytes:
    data = bytearray()
    while chunk := await upload.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Imaginea depaseste 12 MB.")
    return bytes(data)


def _decode_inline_image(encoded: str) -> bytes:
    """Base64 (optionally a data: URL) -> bytes; errors surface as transcription errors."""
    if "," in encoded[:64] and encoded.lstrip().lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        data = base64.b64decode("".join(encoded.split()), validate=True)
    except (binascii.Error, ValueError):
        return b"\x00invalid-base64"  # sniff_image reports "not an image"
    return data


def _form_schema() -> dict:
    return {"type": "object", "properties": {
        "text": {"type": "string", "description": "Blocul brut: intrebarea, apoi A. / B. / C. pe linii separate (sau doar intrebarea)"},
        "image": {"type": "string", "format": "binary", "description": "Poza cu intrebarea (PNG/JPEG/WEBP/GIF, max 12 MB); folosita doar daca textul nu da raspuns"},
        "request_id": {"type": "string"}, "knowledge_base_id": {"type": "string", "default": "default"}}}


def _json_schema() -> dict:
    schema = AnswerRequest.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].rsplit("/", 1)[1]])
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node
    return inline(schema)


@router.post("/answer", response_model=None, openapi_extra={"requestBody": {"required": True, "content": {
    "application/json": {"schema": _json_schema()},
    "multipart/form-data": {"schema": _form_schema()},
    "application/x-www-form-urlencoded": {"schema": _form_schema()}}}},
    responses={200: {"description": "JSON: AnswerResponse (contractul Rokid). Formular: CompactAnswer, sau AnswerResponse cu ?full=1",
                     "model": AnswerResponse}})
async def answer_question(request: Request, full: bool = FULL_PARAM, principal: Principal = Depends(require_device)):
    """Rokid contract (JSON) or quick form (text and/or image). Business errors
    come back as HTTP 200 with `status`; 401/403 token, 413 image too large,
    422 only for an unreadable body."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    image: bytes | None = None
    if content_type == "application/json":
        try:
            raw = await request.json()
        except ValueError:
            raise HTTPException(422, "Corpul nu este JSON valid.")
        if not isinstance(raw, dict):
            raise HTTPException(422, "Corpul JSON trebuie sa fie un obiect.")
        rid = str(raw.get("request_id") or "").strip()[:100]
        try:
            payload = AnswerRequest.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            detail = f"{'.'.join(str(x) for x in first['loc']) or 'body'}: {first['msg']}"
            if not rid:
                raise HTTPException(422, f"Cerere invalida ({detail}).")
            return AnswerResponse(request_id=rid, status="invalid_question", message=f"Cerere invalida ({detail}).")
        kb = payload.knowledge_base_id
        if payload.image is not None:
            image = _decode_inline_image(payload.image.base64)
            if len(image) > MAX_IMAGE_BYTES:
                return AnswerResponse(request_id=rid, status="invalid_question", message="Imaginea depaseste 12 MB.")
        if not payload.has_text() and image is None:
            return AnswerResponse(request_id=rid, status="invalid_question", message="Cererea nu contine nici text, nici imagine.")
        response = await _answer_once(request, principal, rid, kb, payload, image)
        return response

    form = await request.form()
    text = str(form.get("text") or "").strip()
    upload = form.get("image")
    if not isinstance(upload, UploadFile):
        upload = None
    if not text and upload is None:
        raise HTTPException(422, "Trimite campul `text` (blocul cu intrebarea), campul `image` (poza) sau ambele.")
    rid = str(form.get("request_id") or "").strip()[:100] or f"q-{uuid.uuid4().hex[:8]}"
    kb = str(form.get("knowledge_base_id") or "").strip() or "default"
    if upload is not None:
        image = await _read_upload(upload)
    payload = _text_request(rid, kb, text) if text else None
    response = await _answer_once(request, principal, rid, kb, payload, image)
    return response if full else compact(response)
