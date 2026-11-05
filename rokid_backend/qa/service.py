"""Verified Q&A: ingestion, exact and semantic lookup and mapping of stored
correct answers onto the options of a newly observed question."""
from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select

from rokid_backend.core.text import match_key, normalize_text, question_hash, strip_option_label
from rokid_backend.models import IngestionJob, KnowledgeBase, QADataset, QARecord, new_id
from rokid_backend.qa.dataset import DatasetError, QARecordDraft, parse_qa_dataset
from rokid_backend.rag.embeddings import EmbeddingError, EmbeddingProvider
from rokid_backend.rag.vectorstore import VectorStore
from rokid_backend.schemas.api import Option

log = logging.getLogger(__name__)

CONFIRMED_FORMAT = "confirmed"            # dataset holding answers confirmed from admin (no source file; records live in the DB)
CONFIRMED_NAME = "Confirmate din admin"


class QAError(ValueError):
    pass


@dataclass
class OptionMapping:
    labels: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    confident: bool = False
    reason: str = ""


def question_embedding_text(question: str, options: list[str]) -> str:
    """Options are sorted so that a reordered question embeds identically."""
    return normalize_text(question) + " || " + " | ".join(sorted(match_key(strip_option_label(o)) for o in options))


def map_correct_to_options(correct_texts: list[str], options: list[Option], threshold: float, margin: float) -> OptionMapping:
    """Find the label of each stored correct option among the current options by
    text, never by letter. Ambiguity makes the mapping non-confident."""
    if not correct_texts:
        return OptionMapping(reason="inregistrarea nu are raspuns corect")
    if not options:
        return OptionMapping(texts=list(correct_texts), confident=True, reason="intrebare fara optiuni")
    # (label, code-aware key, punctuation-free key, original text)
    current = [(o.label.upper(), match_key(strip_option_label(o.text)), normalize_text(strip_option_label(o.text)), o.text) for o in options]
    labels, texts = [], []
    for correct in correct_texts:
        key, loose = match_key(strip_option_label(correct)), normalize_text(strip_option_label(correct))
        exact = [c for c in current if c[1] == key]
        if not exact and loose:
            exact = [c for c in current if c[2] == loose]
        if len(exact) > 1:
            return OptionMapping(reason=f"mai multe optiuni curente sunt identice cu '{correct[:40]}'")
        if len(exact) == 1:
            chosen = exact[0]
        else:
            scored = sorted(((difflib.SequenceMatcher(None, key, c[1]).ratio(), c) for c in current), key=lambda s: s[0], reverse=True)
            best, second = scored[0], (scored[1] if len(scored) > 1 else None)
            if best[0] < threshold or (second is not None and best[0] - second[0] < margin):
                return OptionMapping(reason=f"optiunea corecta '{correct[:40]}' nu se regaseste clar printre optiunile curente")
            chosen = best[1]
        if chosen[0] in labels:
            return OptionMapping(reason="doua raspunsuri corecte indica aceeasi optiune curenta")
        labels.append(chosen[0])
        texts.append(chosen[3])
    return OptionMapping(labels=labels, texts=texts, confident=True)


class QAService:
    def __init__(self, db, store: VectorStore, embedder: EmbeddingProvider, settings) -> None:
        self.db = db
        self.store = store
        self.embedder = embedder
        self.settings = settings

    # --- datasets -----------------------------------------------------------

    def register_upload(self, kb_id: str, filename: str, data: bytes, name: str = "", anr_category: str | None = None) -> tuple[QADataset, bool]:
        from rokid_backend.core.text import sha256_bytes

        if len(data) > self.settings.max_upload_bytes:
            raise QAError(f"Fisierul depaseste limita de {self.settings.max_upload_mb} MB.")
        if not data:
            raise QAError("Fisierul este gol.")
        filename = Path(filename or "dataset").name[:300]
        ext = Path(filename).suffix.lower()
        if ext not in (".json", ".jsonl", ".csv", ".pdf"):
            raise QAError("Format de set Q&A neacceptat (json, jsonl, csv sau pdf ANR).")
        if ext == ".pdf" and b"%PDF-" not in data[:1024]:
            raise QAError("Fisierul nu este un PDF valid.")
        digest = sha256_bytes(data)
        with self.db.session() as session:
            if session.get(KnowledgeBase, kb_id) is None:
                raise QAError(f"Baza de cunostinte '{kb_id}' nu exista.")
            existing = session.execute(select(QADataset).where(QADataset.kb_id == kb_id, QADataset.content_hash == digest)).scalars().first()
            if existing is not None and existing.status in ("indexed", "indexed_with_errors"):
                return existing, True
            dataset = existing or QADataset(id=new_id("qa"), kb_id=kb_id)
            dataset.name = (name or Path(filename).stem)[:200]
            dataset.filename = filename
            dataset.format = "anr_pdf" if ext == ".pdf" else ext.lstrip(".")
            dataset.content_hash = digest
            dataset.status = "pending"
            dataset.error = ""
            dataset.warnings = [f"anr_category={anr_category.upper()}"] if anr_category else []
            path = self.settings.uploads_dir / kb_id / "qa" / f"{dataset.id}{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            dataset.stored_path = str(path)
            session.add(dataset)
            session.flush()
            return dataset, False

    def index_dataset(self, dataset_id: str, job_id: str | None = None) -> QADataset:
        job_id = job_id or self.create_job(dataset_id)
        stages: dict[str, float] = {}
        started = time.perf_counter()
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            if dataset is None:
                self._finish_job(job_id, "failed", "Setul nu exista.", stages)
                raise QAError("Setul nu exista.")
            dataset.status = "processing"
            kb_id, filename, path = dataset.kb_id, dataset.filename, Path(dataset.stored_path)
            category = next((w.split("=", 1)[1] for w in dataset.warnings if str(w).startswith("anr_category=")), None)
            confirmed = dataset.format == CONFIRMED_FORMAT
            self._update_job(job_id, "running", "Se valideaza inregistrarile...")
        if confirmed:
            return self._reindex_confirmed(dataset_id, kb_id, job_id, stages, started)
        try:
            t = time.perf_counter()
            parsed = parse_qa_dataset(filename, path.read_bytes(), stored_path=path, anr_category=category)
            stages["parse_ms"] = round((time.perf_counter() - t) * 1000, 1)
            t = time.perf_counter()
            embeddings = self.embedder.embed([question_embedding_text(r.question, [o["text"] for o in r.options]) for r in parsed.records])
            stages["embed_ms"] = round((time.perf_counter() - t) * 1000, 1)
            t = time.perf_counter()
            self._replace_records(dataset_id, kb_id, parsed.records, embeddings)
            stages["index_ms"] = round((time.perf_counter() - t) * 1000, 1)
        except (DatasetError, QAError) as exc:
            self._mark_failed(dataset_id, str(exc))
            self._finish_job(job_id, "failed", str(exc), stages)
            return self.get(dataset_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("qa dataset indexing crashed")
            message = str(exc) if isinstance(exc, EmbeddingError) else f"Eroare interna: {type(exc).__name__}"
            self._mark_failed(dataset_id, message)
            self._finish_job(job_id, "failed", message, stages)
            return self.get(dataset_id)
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            dataset.record_count = len(parsed.records)
            dataset.invalid_count = len([e for e in parsed.errors if not e.startswith("avertisment")])
            dataset.status = "indexed_with_errors" if parsed.errors else "indexed"
            keep = [w for w in dataset.warnings if str(w).startswith("anr_category=")]
            dataset.warnings = keep + parsed.errors[:200] + ([f"{parsed.duplicates} duplicate ignorate"] if parsed.duplicates else [])
            dataset.embedding_provider = self.embedder.name
        stages["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._finish_job(job_id, "done", f"{len(parsed.records)} inregistrari indexate, {len(parsed.errors)} probleme.", stages)
        return self.get(dataset_id)

    def _reindex_confirmed(self, dataset_id: str, kb_id: str, job_id: str, stages: dict, started: float) -> QADataset:
        """Confirmed answers have no source file: re-embed the records kept in the database."""
        with self.db.session() as session:
            rows = session.execute(select(QARecord).where(QARecord.dataset_id == dataset_id)).scalars().all()
            items = [(r.id, r.question, [o["text"] for o in r.options]) for r in rows]
        try:
            t = time.perf_counter()
            embeddings = self.embedder.embed([question_embedding_text(q, opts) for _, q, opts in items]) if items else []
            stages["embed_ms"] = round((time.perf_counter() - t) * 1000, 1)
            self.store.delete_where("questions", kb_id, {"dataset_id": dataset_id})
            if items:
                self.store.upsert("questions", kb_id, [i for i, _, _ in items], [q for _, q, _ in items], embeddings,
                                  [{"record_id": i, "dataset_id": dataset_id, "kb_id": kb_id, "kind": "qa", "page": -1} for i, _, _ in items])
        except Exception as exc:  # noqa: BLE001
            log.exception("confirmed dataset reindex crashed")
            message = str(exc) if isinstance(exc, EmbeddingError) else f"Eroare interna: {type(exc).__name__}"
            self._mark_failed(dataset_id, message)
            self._finish_job(job_id, "failed", message, stages)
            return self.get(dataset_id)
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            dataset.status, dataset.record_count, dataset.embedding_provider = "indexed", len(items), self.embedder.name
        stages["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self._finish_job(job_id, "done", f"{len(items)} inregistrari confirmate reindexate.", stages)
        return self.get(dataset_id)

    def _replace_records(self, dataset_id: str, kb_id: str, drafts: list[QARecordDraft], embeddings: list[list[float]]) -> None:
        self.store.delete_where("questions", kb_id, {"dataset_id": dataset_id})
        with self.db.session() as session:
            session.execute(QARecord.__table__.delete().where(QARecord.dataset_id == dataset_id))
            ids, texts, metas = [], [], []
            for draft in drafts:
                record = QARecord(id=new_id("rec"), dataset_id=dataset_id, kb_id=kb_id, question=draft.question,
                                  normalized_question=draft.normalized_question, question_hash=draft.question_hash,
                                  question_type=draft.question_type, options=draft.options,
                                  correct_labels=draft.correct_labels, correct_texts=draft.correct_texts,
                                  explanation=draft.explanation, source=draft.source)
                session.add(record)
                ids.append(record.id)
                texts.append(draft.question)
                metas.append({"record_id": record.id, "dataset_id": dataset_id, "kb_id": kb_id, "kind": "qa",
                              "page": int(draft.source.get("page", -1) or -1)})
            session.flush()
        self.store.upsert("questions", kb_id, ids, texts, embeddings, metas)

    def reindex(self, dataset_id: str) -> str:
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            if dataset is None:
                raise QAError("Setul nu exista.")
            if dataset.format != CONFIRMED_FORMAT and not Path(dataset.stored_path).exists():
                raise QAError("Fisierul original lipseste; incarca setul din nou.")
            dataset.status = "pending"
        return self.create_job(dataset_id)

    def delete(self, dataset_id: str) -> None:
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            if dataset is None:
                raise QAError("Setul nu exista.")
            kb_id, path = dataset.kb_id, dataset.stored_path
            session.delete(dataset)
        self.store.delete_where("questions", kb_id, {"dataset_id": dataset_id})
        if path:
            Path(path).unlink(missing_ok=True)

    def get(self, dataset_id: str) -> QADataset:
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            if dataset is None:
                raise QAError("Setul nu exista.")
            return dataset

    # --- lookup -------------------------------------------------------------

    def exact_matches(self, kb_id: str, question: str) -> list[QARecord]:
        with self.db.session() as session:
            rows = session.execute(select(QARecord).where(QARecord.kb_id == kb_id, QARecord.question_hash == question_hash(question))).scalars().all()
            return list(rows)

    def semantic_matches(self, kb_id: str, embedding: list[float], k: int) -> list[tuple[QARecord, float]]:
        hits = self.store.query("questions", kb_id, embedding, k)
        if not hits:
            return []
        ids = [h.metadata.get("record_id") for h in hits]
        with self.db.session() as session:
            records = {r.id: r for r in session.execute(select(QARecord).where(QARecord.id.in_(ids))).scalars()}
        return [(records[i], h.similarity) for i, h in zip(ids, hits) if i in records]

    def dataset_names(self, dataset_ids: set[str]) -> dict[str, QADataset]:
        if not dataset_ids:
            return {}
        with self.db.session() as session:
            return {d.id: d for d in session.execute(select(QADataset).where(QADataset.id.in_(dataset_ids))).scalars()}

    def record_count(self, kb_id: str) -> int:
        with self.db.session() as session:
            return session.execute(select(func.count()).select_from(QARecord).where(QARecord.kb_id == kb_id)).scalar() or 0

    # --- jobs ---------------------------------------------------------------

    def create_job(self, dataset_id: str) -> str:
        with self.db.session() as session:
            job = IngestionJob(id=new_id("job"), kind="qa_dataset", target_id=dataset_id, status="queued")
            session.add(job)
            return job.id

    def _mark_failed(self, dataset_id: str, message: str) -> None:
        with self.db.session() as session:
            dataset = session.get(QADataset, dataset_id)
            if dataset:
                dataset.status, dataset.error = "failed", message[:2000]

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
