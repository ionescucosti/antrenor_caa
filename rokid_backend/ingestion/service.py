"""Document ingestion: store the file, parse, chunk, embed, index in ChromaDB
and record status/errors. Also reindex and delete."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from sqlalchemy import select

from rokid_backend.core.text import sha256_bytes
from rokid_backend.ingestion.chunking import chunk_blocks
from rokid_backend.ingestion.parsers import ParseError, parse_document, sniff_type
from rokid_backend.models import Document, IngestionJob, KnowledgeBase, new_id
from rokid_backend.rag.embeddings import EmbeddingProvider
from rokid_backend.rag.vectorstore import VectorStore

log = logging.getLogger(__name__)


class IngestionError(ValueError):
    pass


class DocumentService:
    def __init__(self, db, store: VectorStore, embedder: EmbeddingProvider, settings) -> None:
        self.db = db
        self.store = store
        self.embedder = embedder
        self.settings = settings

    # --- registration -------------------------------------------------------

    def register_upload(self, kb_id: str, filename: str, data: bytes, content_type: str = "",
                        replace_document_id: str | None = None) -> tuple[Document, bool]:
        """Store the file and create (or bump) the Document row. Returns
        (document, duplicate) where duplicate means identical content already
        indexed in the same knowledge base, so no re-indexing is required."""
        if len(data) > self.settings.max_upload_bytes:
            raise IngestionError(f"Fisierul depaseste limita de {self.settings.max_upload_mb} MB.")
        if not data:
            raise IngestionError("Fisierul este gol.")
        filename = Path(filename or "upload").name[:300]
        try:
            sniff_type(filename, data)
        except ParseError as exc:
            raise IngestionError(str(exc)) from exc
        digest = sha256_bytes(data)
        with self.db.session() as session:
            if session.get(KnowledgeBase, kb_id) is None:
                raise IngestionError(f"Baza de cunostinte '{kb_id}' nu exista.")
            if replace_document_id:
                doc = session.get(Document, replace_document_id)
                if doc is None or doc.kb_id != kb_id:
                    raise IngestionError("Documentul de inlocuit nu exista in aceasta baza de cunostinte.")
                if doc.content_hash == digest and doc.status == "indexed":
                    return doc, True
                doc.version += 1
                doc.filename = filename
            else:
                existing = session.execute(select(Document).where(
                    Document.kb_id == kb_id, Document.content_hash == digest)).scalars().first()
                if existing is not None and existing.status == "indexed":
                    return existing, True
                doc = existing or Document(id=new_id("doc"), kb_id=kb_id, version=1)
                doc.filename = filename
                session.add(doc)
            doc.content_hash = digest
            doc.content_type = content_type[:100]
            doc.size_bytes = len(data)
            doc.status = "pending"
            doc.error = ""
            doc.warnings = []
            path = self.settings.uploads_dir / kb_id / f"{doc.id}-v{doc.version}{Path(filename).suffix.lower()}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            doc.stored_path = str(path)
            session.flush()
            return doc, False

    # --- indexing -----------------------------------------------------------

    def index_document(self, document_id: str, job_id: str | None = None) -> Document:
        job_id = job_id or self.create_job("document", document_id)
        started = time.perf_counter()
        stages: dict[str, float] = {}
        with self.db.session() as session:
            doc = session.get(Document, document_id)
            if doc is None:
                self._finish_job(job_id, "failed", "Documentul nu exista.", stages)
                raise IngestionError("Documentul nu exista.")
            doc.status = "processing"
            doc.error = ""
            path = Path(doc.stored_path)
            kb_id, filename, version = doc.kb_id, doc.filename, doc.version
            self._update_job(job_id, "running", "Se analizeaza documentul...")
        try:
            data = path.read_bytes()
            t = time.perf_counter()
            parsed = parse_document(filename, data, ocr_enabled=self.settings.ocr_enabled,
                                    ocr_language=self.settings.ocr_language)
            stages["parse_ms"] = round((time.perf_counter() - t) * 1000, 1)
            t = time.perf_counter()
            chunks = chunk_blocks(parsed.blocks, self.settings.chunk_size, self.settings.chunk_overlap)
            if not chunks:
                raise ParseError("Nu a rezultat niciun fragment indexabil.")
            stages["chunk_ms"] = round((time.perf_counter() - t) * 1000, 1)
            t = time.perf_counter()
            embeddings = self.embedder.embed([c.text for c in chunks])
            stages["embed_ms"] = round((time.perf_counter() - t) * 1000, 1)
            t = time.perf_counter()
            self.store.delete_where("chunks", kb_id, {"document_id": document_id})
            ids = [f"{document_id}:{c.index}" for c in chunks]
            metadatas = [{
                "document_id": document_id, "document_name": filename, "kb_id": kb_id,
                "page": c.page if c.page is not None else -1, "section": c.section or "",
                "chunk_index": c.index, "version": version, "content_hash": sha256_bytes(c.text.encode("utf-8")),
                "kind": "document",
            } for c in chunks]
            self.store.upsert("chunks", kb_id, ids, [c.text for c in chunks], embeddings, metadatas)
            stages["index_ms"] = round((time.perf_counter() - t) * 1000, 1)
        except (ParseError, IngestionError) as exc:
            self._mark_failed(document_id, str(exc))
            self._finish_job(job_id, "failed", str(exc), stages)
            return self.get(document_id)
        except Exception as exc:  # noqa: BLE001 - any parser crash must surface as a failed job
            log.exception("document indexing crashed", extra={"document_id": document_id})
            message = f"Eroare interna la indexare: {type(exc).__name__}"
            self._mark_failed(document_id, message)
            self._finish_job(job_id, "failed", message, stages)
            return self.get(document_id)
        with self.db.session() as session:
            doc = session.get(Document, document_id)
            doc.status = "indexed"
            doc.pages = parsed.pages
            doc.chunk_count = len(chunks)
            doc.warnings = parsed.warnings[:50] + (["Text obtinut prin OCR."] if parsed.used_ocr else [])
            doc.embedding_provider = self.embedder.name
            doc.indexed_at = time.time()
        stages["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._finish_job(job_id, "done", f"{len(chunks)} fragmente indexate.", stages)
        return self.get(document_id)

    def reindex(self, document_id: str) -> str:
        with self.db.session() as session:
            doc = session.get(Document, document_id)
            if doc is None:
                raise IngestionError("Documentul nu exista.")
            if not Path(doc.stored_path).exists():
                raise IngestionError("Fisierul original lipseste de pe disc; incarca documentul din nou.")
            doc.status = "pending"
        return self.create_job("document", document_id)

    def delete(self, document_id: str) -> None:
        with self.db.session() as session:
            doc = session.get(Document, document_id)
            if doc is None:
                raise IngestionError("Documentul nu exista.")
            kb_id, path = doc.kb_id, doc.stored_path
            session.delete(doc)
        self.store.delete_where("chunks", kb_id, {"document_id": document_id})
        if path:
            Path(path).unlink(missing_ok=True)

    # --- helpers ------------------------------------------------------------

    def get(self, document_id: str) -> Document:
        with self.db.session() as session:
            doc = session.get(Document, document_id)
            if doc is None:
                raise IngestionError("Documentul nu exista.")
            return doc

    def _mark_failed(self, document_id: str, message: str) -> None:
        with self.db.session() as session:
            doc = session.get(Document, document_id)
            if doc is not None:
                doc.status = "failed"
                doc.error = message[:2000]

    def create_job(self, kind: str, target_id: str) -> str:
        with self.db.session() as session:
            job = IngestionJob(id=new_id("job"), kind=kind, target_id=target_id, status="queued")
            session.add(job)
            return job.id

    def _update_job(self, job_id: str, status: str, message: str) -> None:
        with self.db.session() as session:
            job = session.get(IngestionJob, job_id)
            if job:
                job.status, job.message = status, message[:2000]

    def _finish_job(self, job_id: str, status: str, message: str, stages: dict) -> None:
        with self.db.session() as session:
            job = session.get(IngestionJob, job_id)
            if job:
                job.status, job.message, job.stages, job.finished_at = status, message[:2000], stages, time.time()
