"""Question answering orchestration.

Precedence: exact verified Q&A -> semantically matching verified Q&A ->
general document retrieval + LLM. Strict document grounding by default."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from rokid_backend.core.rawtext import split_raw_question
from rokid_backend.core.text import letters_ratio, match_key, normalize_text, strip_option_label
from rokid_backend.core.timing import StageTimer
from rokid_backend.llm.base import ANSWER_SCHEMA, LLMError, parse_answer
from rokid_backend.models import QARecord
from rokid_backend.qa.service import OptionMapping, QAService, map_correct_to_options, question_embedding_text
from rokid_backend.rag.prompts import SYSTEM_PROMPT, Evidence, build_user_prompt
from rokid_backend.rag.vectorstore import VectorStore
from rokid_backend.schemas.api import Answer, AnswerRequest, AnswerResponse, Option, SourceRef

log = logging.getLogger(__name__)


@dataclass
class VerifiedCandidate:
    record: QARecord
    similarity: float
    mapping: OptionMapping


class AnswerService:
    def __init__(self, qa: QAService, store: VectorStore, embedder, llm, settings, kb_exists) -> None:
        self.qa = qa
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.settings = settings
        self.kb_exists = kb_exists

    # --- public -------------------------------------------------------------

    def answer(self, req: AnswerRequest) -> AnswerResponse:
        timer = StageTimer()
        with timer.stage("validation"):
            req = self._structure(req)
            problem = self._validate(req)
        if problem:
            return self._finish(req, timer, "invalid_question", message=problem)
        if not self.kb_exists(req.knowledge_base_id):
            return self._finish(req, timer, "invalid_question", message=f"Baza de cunostinte '{req.knowledge_base_id}' nu exista.")
        has_options = bool(req.options)

        # 1. exact verified match
        with timer.stage("exact_match"):
            exact = self.qa.exact_matches(req.knowledge_base_id, req.question)
            candidates = [VerifiedCandidate(r, 1.0, self._map(r, req)) for r in exact]
        if candidates:
            outcome = self._decide_verified(candidates, req, has_options, require_all=True)
            if outcome is not None:
                status, answer, sources, message = outcome
                return self._finish(req, timer, status, answer, sources, message, "exact_qa" if status == "answered" else "none")

        # 2. semantic verified match
        with timer.stage("embedding"):
            vector = self.embedder.embed([question_embedding_text(req.question, [o.text for o in req.options])])[0]
        with timer.stage("semantic_qa"):
            semantic = [VerifiedCandidate(r, s, self._map(r, req)) for r, s in
                        self.qa.semantic_matches(req.knowledge_base_id, vector, self.settings.top_k_questions)]
            strong = [c for c in semantic if c.similarity >= self.settings.semantic_question_threshold]
        if strong:
            outcome = self._decide_verified(strong, req, has_options, require_all=False)
            if outcome is not None and outcome[0] == "answered":
                status, answer, sources, message = outcome
                return self._finish(req, timer, status, answer, sources, message, "semantic_qa")

        # 3. document retrieval
        with timer.stage("vector_retrieval"):
            hits = self.store.query("chunks", req.knowledge_base_id, vector, self.settings.top_k_chunks)
            hits = [h for h in hits if h.similarity >= self.settings.min_chunk_similarity]
        evidence: list[Evidence] = []
        datasets = self.qa.dataset_names({c.record.dataset_id for c in semantic})
        for cand in semantic:
            if cand.similarity < self.settings.semantic_question_evidence_floor:
                continue
            ds = datasets.get(cand.record.dataset_id)
            evidence.append(Evidence(
                id=f"S{len(evidence) + 1}", kind="qa", title=f"{ds.name if ds else 'set Q&A'} (verified)",
                text=self._record_as_text(cand.record), similarity=cand.similarity,
                metadata={"document_id": cand.record.dataset_id, "document_name": (ds.filename or ds.name) if ds else "qa",
                          "page": cand.record.source.get("page"), "section": None, "kind": "qa_dataset"}))
        for hit in hits:
            page = hit.metadata.get("page")
            evidence.append(Evidence(
                id=f"S{len(evidence) + 1}", kind="document",
                title=f"{hit.metadata.get('document_name', 'document')}" + (f" p.{page}" if isinstance(page, int) and page > 0 else ""),
                text=hit.text, similarity=hit.similarity,
                metadata={"document_id": hit.metadata.get("document_id", ""), "document_name": hit.metadata.get("document_name", ""),
                          "page": page if isinstance(page, int) and page > 0 else None,
                          "section": hit.metadata.get("section") or None, "kind": "document"}))
        timer.mark("reranking", 0.0)

        general = self.settings.allow_general_knowledge and not evidence
        if not evidence and not general:
            return self._finish(req, timer, "insufficient_context",
                                message="Nu exista dovezi relevante in baza de cunostinte pentru aceasta intrebare.")

        # 4. LLM over evidence
        options = [(o.label.upper(), o.text) for o in req.options]
        user_prompt = build_user_prompt(req.question, req.question_type, options, evidence, general_knowledge=general)
        try:
            with timer.stage("llm"):
                raw = self.llm.generate(SYSTEM_PROMPT + (("\n" + "General-knowledge mode may be enabled below.") if general else ""), user_prompt, ANSWER_SCHEMA)
        except LLMError as exc:
            log.warning("llm failure: %s", exc)
            return self._finish(req, timer, "error", message=f"Generarea raspunsului a esuat: {exc}")
        with timer.stage("answer_validation"):
            try:
                parsed = parse_answer(raw)
            except LLMError as exc:
                return self._finish(req, timer, "error", message=str(exc))
            status, answer, sources, message = self._validate_llm_answer(parsed, req, evidence, general)
        return self._finish(req, timer, status, answer, sources, message,
                            ("general_knowledge" if general else "rag") if status == "answered" else "none")

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _structure(req: AnswerRequest) -> AnswerRequest:
        """Requests without options may carry the whole OCR block (question plus
        'A. ...' lines) in question or raw_ocr_text: split it here."""
        if req.options:
            return req
        for source in (req.question, req.raw_ocr_text or ""):
            question, options = split_raw_question(source)
            if options:
                return req.model_copy(update={"question": question, "options": [Option(label=l, text=t) for l, t in options],
                                              "question_type": "multiple_choice" if req.question_type == "open" else req.question_type})
        if not req.question.strip() and req.raw_ocr_text:
            return req.model_copy(update={"question": req.raw_ocr_text.strip()})
        return req

    @staticmethod
    def _validate(req: AnswerRequest) -> str | None:
        question = normalize_text(req.question)
        if len(question) < 8:
            return "Intrebarea este prea scurta sau textul OCR este incomplet."
        if letters_ratio(req.question) < 0.5:
            return "Textul intrebarii pare deteriorat (prea putine litere)."
        if len(req.options) == 1:
            return "O singura optiune de raspuns: textul OCR pare incomplet."
        texts = [match_key(strip_option_label(o.text)) for o in req.options]
        if any(not t for t in texts):
            return "Exista optiuni de raspuns fara text."
        if len(set(texts)) != len(texts):
            return "Optiunile de raspuns contin duplicate; textul OCR pare deteriorat."
        return None

    def _map(self, record: QARecord, req: AnswerRequest) -> OptionMapping:
        return map_correct_to_options(list(record.correct_texts), req.options,
                                      self.settings.option_match_threshold, self.settings.option_match_margin)

    def _decide_verified(self, candidates: list[VerifiedCandidate], req: AnswerRequest, has_options: bool, require_all: bool):
        """Returns (status, answer, sources, message) or None to fall through."""
        confident = [c for c in candidates if c.mapping.confident]
        if not confident:
            if require_all:
                reasons = "; ".join(sorted({c.mapping.reason for c in candidates}))
                return ("insufficient_context", None, self._sources(candidates), f"Intrebare cunoscuta, dar raspunsul nu poate fi mapat sigur pe optiunile curente: {reasons}.")
            return None
        keys = {tuple(sorted(c.mapping.labels)) if has_options else tuple(sorted(match_key(t) for t in c.mapping.texts)) for c in confident}
        if len(keys) > 1:
            if require_all:
                return ("insufficient_context", None, self._sources(confident), "Inregistrarile verificate sunt in conflict pentru aceasta intrebare.")
            # semantic: trust the best only if clearly ahead of a conflicting one
            best = max(confident, key=lambda c: c.similarity)
            rivals = [c for c in confident if c is not best and (tuple(sorted(c.mapping.labels)) != tuple(sorted(best.mapping.labels)))]
            if any(best.similarity - r.similarity < 0.05 for r in rivals):
                return None
            confident = [best]
        best = max(confident, key=lambda c: c.similarity)
        explanation = best.record.explanation or "Raspuns verificat din setul de intrebari cunoscute."
        if best.similarity < 1.0:
            explanation = f"{explanation} (intrebare similara, {best.similarity:.0%})"
        answer = Answer(option_labels=best.mapping.labels if has_options else [],
                        text="; ".join(best.mapping.texts), explanation=explanation[:600])
        return ("answered", answer, self._sources(confident), "")

    def _sources(self, candidates: list[VerifiedCandidate]) -> list[SourceRef]:
        datasets = self.qa.dataset_names({c.record.dataset_id for c in candidates})
        refs, seen = [], set()
        for c in candidates:
            ds = datasets.get(c.record.dataset_id)
            page = c.record.source.get("page")
            key = (c.record.dataset_id, page)
            if key in seen:
                continue
            seen.add(key)
            refs.append(SourceRef(document_id=c.record.dataset_id, document_name=(ds.filename or ds.name) if ds else "qa_dataset",
                                  page=page if isinstance(page, int) else None, kind="qa_dataset", score=round(c.similarity, 3),
                                  section=(f"nr. {c.record.source.get('number')}" if c.record.source.get("number") else None)))
        return refs

    @staticmethod
    def _record_as_text(record: QARecord) -> str:
        parts = [f"Question: {record.question}"]
        if record.options:
            parts.append("Options: " + "; ".join(f"{o['label']}. {o['text']}" for o in record.options))
        parts.append("Correct answer: " + "; ".join(record.correct_texts))
        if record.explanation:
            parts.append(f"Explanation: {record.explanation}")
        return "\n".join(parts)

    def _validate_llm_answer(self, parsed, req: AnswerRequest, evidence: list[Evidence], general: bool):
        allowed_labels = {o.label.upper() for o in req.options}
        allowed_ids = {e.id for e in evidence}
        cited = [i for i in parsed.evidence_ids if i in allowed_ids]
        if parsed.status != "answered":
            return ("insufficient_context", None, self._sources_from_evidence(evidence, cited), parsed.explanation or "Dovezile disponibile nu sustin un raspuns.")
        labels = [l for l in parsed.option_labels if l in allowed_labels]
        if req.options and (not labels or len(labels) != len(parsed.option_labels)):
            return ("insufficient_context", None, self._sources_from_evidence(evidence, cited), "Modelul a indicat optiuni care nu exista in intrebare.")
        if not req.options and not parsed.answer_text.strip():
            return ("insufficient_context", None, [], "Modelul nu a produs un raspuns.")
        if not cited and not general:
            return ("insufficient_context", None, [], "Raspunsul nu a fost sustinut de nicio sursa citata; politica stricta de fundamentare l-a respins.")
        by_label = {o.label.upper(): o.text for o in req.options}
        text = "; ".join(by_label[l] for l in labels) if labels else parsed.answer_text.strip()
        explanation = parsed.explanation.strip()
        if general and not cited:
            explanation = (explanation + " (fara surse din documentele incarcate)").strip()
        return ("answered", Answer(option_labels=labels, text=text[:1000], explanation=explanation[:600]),
                self._sources_from_evidence(evidence, cited), "")

    @staticmethod
    def _sources_from_evidence(evidence: list[Evidence], cited: list[str]) -> list[SourceRef]:
        refs, seen = [], set()
        for item in evidence:
            if item.id not in cited:
                continue
            m = item.metadata
            key = (m.get("document_id"), m.get("page"))
            if key in seen:
                continue
            seen.add(key)
            refs.append(SourceRef(document_id=str(m.get("document_id") or ""), document_name=str(m.get("document_name") or ""),
                                  page=m.get("page"), section=m.get("section"), kind=m.get("kind", "document"), score=item.similarity))
        return refs

    @staticmethod
    def _finish(req: AnswerRequest, timer: StageTimer, status: str, answer: Answer | None = None,
                sources: list[SourceRef] | None = None, message: str = "", match_type: str = "none") -> AnswerResponse:
        with timer.stage("serialization"):
            response = AnswerResponse(request_id=req.request_id, status=status, answer=answer, sources=sources or [],
                                      message=message, match_type=match_type)
        response.timings_ms = dict(timer.stages)
        response.processing_time_ms = timer.total_ms
        return response
