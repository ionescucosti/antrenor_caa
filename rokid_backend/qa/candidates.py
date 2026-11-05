"""Answers that did not come from a verified set (documentation, model
knowledge, web search) are kept as candidates. Confirming one in admin turns it
into a verified QARecord in the knowledge base's dataset "Confirmate din
admin", so the same question is answered from the verified set from then on."""
from __future__ import annotations

import logging
import time

from sqlalchemy import func, select

from rokid_backend.core.text import normalize_text, question_hash
from rokid_backend.models import AnswerCandidate, QADataset, QARecord, new_id
from rokid_backend.qa.service import CONFIRMED_FORMAT, CONFIRMED_NAME, question_embedding_text
from rokid_backend.schemas.api import Option, SourceRef

log = logging.getLogger(__name__)
_FINAL = ("pending", "confirmed", "rejected")


class CandidateError(ValueError):
    pass


class CandidateService:
    def __init__(self, db, store, embedder, settings) -> None:
        self.db = db
        self.store = store
        self.embedder = embedder
        self.settings = settings

    # --- recording ----------------------------------------------------------

    def record(self, kb_id: str, request_id: str, question: str, options: list[Option], labels: list[str], text: str,
               explanation: str, agreement: str, match_type: str, docs_labels: list[str], web_labels: list[str],
               sources: list[SourceRef]) -> str:
        """Insert a pending candidate, or refresh the pending one for the same
        question (several frames of the same screen must not pile up)."""
        digest = question_hash(question)
        opts = [{"label": o.label.upper(), "text": o.text} for o in options]
        srcs = [s.model_dump() for s in sources][:12]
        with self.db.session() as session:
            cand = session.execute(select(AnswerCandidate).where(AnswerCandidate.kb_id == kb_id, AnswerCandidate.question_hash == digest,
                                                                 AnswerCandidate.status == "pending")).scalars().first()
            if cand is None:
                cand = AnswerCandidate(id=new_id("cand"), kb_id=kb_id, question=question, normalized_question=normalize_text(question),
                                       question_hash=digest, seen_count=0)
                session.add(cand)
            cand.seen_count = (cand.seen_count or 0) + 1
            cand.options, cand.option_labels, cand.answer_text = opts, list(labels), (text or "")[:2000]
            cand.explanation, cand.agreement, cand.match_type = (explanation or "")[:2000], agreement, match_type
            cand.docs_labels, cand.web_labels, cand.sources, cand.request_id = list(docs_labels), list(web_labels), srcs, request_id[:100]
            session.flush()
            return cand.id

    # --- review -------------------------------------------------------------

    def list(self, kb_id: str | None = None, status: str = "pending", limit: int = 50) -> list[AnswerCandidate]:
        with self.db.session() as session:
            query = select(AnswerCandidate).order_by(AnswerCandidate.updated_at.desc()).limit(max(1, min(limit, 500)))
            if kb_id:
                query = query.where(AnswerCandidate.kb_id == kb_id)
            if status and status != "all":
                query = query.where(AnswerCandidate.status == status)
            return list(session.execute(query).scalars())

    def pending_count(self, kb_id: str | None = None) -> int:
        with self.db.session() as session:
            query = select(func.count()).select_from(AnswerCandidate).where(AnswerCandidate.status == "pending")
            if kb_id:
                query = query.where(AnswerCandidate.kb_id == kb_id)
            return session.execute(query).scalar() or 0

    def get(self, candidate_id: str) -> AnswerCandidate:
        with self.db.session() as session:
            cand = session.get(AnswerCandidate, candidate_id)
            if cand is None:
                raise CandidateError("Candidatul nu exista.")
            return cand

    def reject(self, candidate_id: str) -> None:
        with self.db.session() as session:
            cand = session.get(AnswerCandidate, candidate_id)
            if cand is None:
                raise CandidateError("Candidatul nu exista.")
            cand.status = "rejected"

    def confirm(self, candidate_id: str, labels: list[str] | None = None, explanation: str = "") -> str:
        """Create (or update) the verified record for this question and index
        its embedding; returns the QARecord id."""
        with self.db.session() as session:
            cand = session.get(AnswerCandidate, candidate_id)
            if cand is None:
                raise CandidateError("Candidatul nu exista.")
            chosen = [str(l).strip().upper() for l in (labels if labels is not None else cand.option_labels) if str(l).strip()]
            by_label = {str(o["label"]).upper(): str(o["text"]) for o in cand.options}
            if cand.options:
                if not chosen or any(l not in by_label for l in chosen):
                    raise CandidateError("Alege o optiune existenta (A, B, C...).")
                texts = [by_label[l] for l in chosen]
            else:
                texts = [cand.answer_text] if (cand.answer_text or "").strip() else []
                chosen = []
                if not texts:
                    raise CandidateError("Candidatul nu are un raspuns de confirmat.")
            dataset = self._confirmed_dataset(session, cand.kb_id)
            record = session.execute(select(QARecord).where(QARecord.dataset_id == dataset.id,
                                                            QARecord.question_hash == cand.question_hash)).scalars().first()
            if record is None:
                record = QARecord(id=new_id("rec"), dataset_id=dataset.id, kb_id=cand.kb_id, question=cand.question,
                                  normalized_question=normalize_text(cand.question), question_hash=cand.question_hash)
                session.add(record)
                dataset.record_count = (dataset.record_count or 0) + 1
            record.question_type = "multiple_choice" if cand.options else "open"
            record.options, record.correct_labels, record.correct_texts = list(cand.options), chosen, texts
            record.explanation = (explanation or cand.explanation or "")[:2000]
            record.source = {"kind": "confirmed", "candidate_id": cand.id, "request_id": cand.request_id,
                             "agreement": cand.agreement, "confirmed_at": time.time()}
            dataset.embedding_provider = self.embedder.name
            cand.status, cand.record_id, cand.option_labels = "confirmed", record.id, chosen
            session.flush()
            rec_id, ds_id, kb_id, question = record.id, dataset.id, cand.kb_id, cand.question
            option_texts = [str(o["text"]) for o in cand.options]
        embedding = self.embedder.embed([question_embedding_text(question, option_texts)])
        self.store.upsert("questions", kb_id, [rec_id], [question], embedding,
                          [{"record_id": rec_id, "dataset_id": ds_id, "kb_id": kb_id, "kind": "qa", "page": -1}])
        log.info("candidate confirmed", extra={"candidate_id": candidate_id, "record_id": rec_id})
        return rec_id

    @staticmethod
    def _confirmed_dataset(session, kb_id: str) -> QADataset:
        dataset = session.execute(select(QADataset).where(QADataset.kb_id == kb_id, QADataset.format == CONFIRMED_FORMAT)).scalars().first()
        if dataset is None:
            dataset = QADataset(id=new_id("qa"), kb_id=kb_id, name=CONFIRMED_NAME, filename="", format=CONFIRMED_FORMAT,
                                content_hash=f"confirmed:{kb_id}", status="indexed", stored_path="")
            session.add(dataset)
            session.flush()
        return dataset
