"""Question answering orchestration.

Precedence: exact verified Q&A -> semantically matching verified Q&A -> for an
unknown question, the documentation answer (retrieval + LLM) and the web search
answer run in parallel and are combined: agreement is shown as is, disagreement
is shown marked ("Verifica: ..."), and every non-verified answer is kept as a
candidate that admin can confirm into the verified set."""
from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from dataclasses import dataclass, field

from rokid_backend.core.rawtext import split_raw_question
from rokid_backend.core.text import letters_ratio, match_key, normalize_text, strip_option_label
from rokid_backend.core.timing import StageTimer
from rokid_backend.llm.base import ANSWER_SCHEMA, LLMError, parse_answer
from rokid_backend.llm.web_search import domain_of
from rokid_backend.models import QARecord
from rokid_backend.qa.service import OptionMapping, QAService, map_correct_to_options, question_embedding_text
from rokid_backend.rag.embeddings import EmbeddingError
from rokid_backend.rag.prompts import Evidence, build_system_prompt, build_user_prompt, build_web_instructions
from rokid_backend.rag.vectorstore import VectorStore
from rokid_backend.schemas.api import Answer, AnswerRequest, AnswerResponse, Option, SourceRef

log = logging.getLogger(__name__)
_BRANCH_NAMES = {"docs": "documentatia", "web": "internetul"}


@dataclass
class VerifiedCandidate:
    record: QARecord
    similarity: float
    mapping: OptionMapping


@dataclass
class Branch:
    """Outcome of one answering branch for an unknown question."""
    kind: str                                   # docs | web
    status: str = "insufficient"                # answered | insufficient | error | timeout
    labels: list[str] = field(default_factory=list)
    text: str = ""
    explanation: str = ""
    sources: list[SourceRef] = field(default_factory=list)
    message: str = ""
    general: bool = False                       # docs: answered without citing uploaded evidence
    multi: bool = False                         # several labels on a single-answer question: not trusted
    ms: int = 0
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status == "answered" and not self.multi


class AnswerService:
    def __init__(self, qa: QAService, store: VectorStore, embedder, llm, settings, kb_exists, web=None, candidates=None) -> None:
        self.qa = qa
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.settings = settings
        self.kb_exists = kb_exists
        self.web = web                      # WebSearchProvider or None
        self.candidates = candidates        # CandidateService or None
        self._pool = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="answer")

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
                return self._finish(req, timer, status, answer, sources, message, "exact_qa" if status == "answered" else "none",
                                    "verified" if status == "answered" else "none")

        # 2. semantic verified match
        with timer.stage("embedding"):
            try:
                vector = self.embedder.embed([question_embedding_text(req.question, [o.text for o in req.options])])[0]
            except Exception as exc:  # noqa: BLE001 - network, quota, provider errors
                log.warning("embedding failure: %s", exc)
                detail = str(exc) if isinstance(exc, EmbeddingError) else type(exc).__name__
                return self._finish(req, timer, "error", message=f"Calculul embeddings a esuat: {detail}")
        with timer.stage("semantic_qa"):
            semantic = [VerifiedCandidate(r, s, self._map(r, req)) for r, s in
                        self.qa.semantic_matches(req.knowledge_base_id, vector, self.settings.top_k_questions)]
            strong = [c for c in semantic if c.similarity >= self.settings.semantic_question_threshold]
        if strong:
            outcome = self._decide_verified(strong, req, has_options, require_all=False)
            if outcome is not None and outcome[0] == "answered":
                status, answer, sources, message = outcome
                return self._finish(req, timer, status, answer, sources, message, "semantic_qa", "verified")

        # 3. unknown question: documentation and web search in parallel
        started = time.perf_counter()
        docs_future = self._pool.submit(self._docs_answer, req, vector, semantic)
        web_future = self._pool.submit(self._web_answer, req) if self.web is not None else None
        docs = self._wait(docs_future, started + self.settings.docs_timeout_seconds, "docs")
        web = self._wait(web_future, started + self.settings.web_search_timeout_seconds, "web") if web_future is not None else None
        timer.mark("parallel", (time.perf_counter() - started) * 1000)
        for branch in (docs, web):
            if branch is not None:
                timer.mark(f"{branch.kind}_ms", branch.ms)
                for name, ms in branch.stages.items():
                    timer.mark(name, ms)
        status, answer, sources, message, match_type, agreement = self._decide_unknown(docs, web)
        response = self._finish(req, timer, status, answer, sources, message, match_type, agreement)
        if status == "answered" and answer is not None and self.candidates is not None and self.settings.candidates_enabled:
            try:
                response.candidate_id = self.candidates.record(
                    req.knowledge_base_id, req.request_id, req.question, req.options, answer.option_labels, answer.text,
                    answer.explanation, agreement, match_type, docs.labels if docs.status == "answered" else [],
                    web.labels if web is not None and web.status == "answered" else [], sources)
            except Exception:  # noqa: BLE001 - reviewing is optional, answering is not
                log.exception("could not record the answer candidate")
        return response

    # --- unknown questions ------------------------------------------------

    @staticmethod
    def _wait(future, deadline: float, kind: str) -> Branch:
        try:
            return future.result(timeout=max(0.0, deadline - time.perf_counter()))
        except cf.TimeoutError:
            log.warning("%s answer timed out", kind)
            return Branch(kind=kind, status="timeout", message=f"Raspunsul din {_BRANCH_NAMES[kind]} a depasit timpul alocat.")
        except Exception as exc:  # noqa: BLE001
            log.exception("%s answer crashed", kind)
            return Branch(kind=kind, status="error", message=f"Eroare interna in {_BRANCH_NAMES[kind]} ({type(exc).__name__}).")

    def _evidence(self, req: AnswerRequest, semantic: list[VerifiedCandidate], hits) -> list[Evidence]:
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
        return evidence

    def _docs_answer(self, req: AnswerRequest, vector: list[float], semantic: list[VerifiedCandidate]) -> Branch:
        """Retrieval over the uploaded documents (plus loosely similar verified
        records as evidence), then the LLM. Strict grounding abstains without
        evidence; open grounding lets the model use its own knowledge."""
        branch, timer, started = Branch(kind="docs"), StageTimer(), time.perf_counter()
        try:
            with timer.stage("vector_retrieval"):
                hits = self.store.query("chunks", req.knowledge_base_id, vector, self.settings.top_k_chunks)
                hits = [h for h in hits if h.similarity >= self.settings.min_chunk_similarity]
            evidence = self._evidence(req, semantic, hits)
            timer.mark("reranking", 0.0)
            strict = self.settings.answer_grounding != "open"
            general = strict and self.settings.allow_general_knowledge and not evidence
            if strict and not evidence and not general:
                branch.message = "Nu exista dovezi relevante in baza de cunostinte pentru aceasta intrebare."
                return branch
            options = [(o.label.upper(), o.text) for o in req.options]
            user_prompt = build_user_prompt(req.question, req.question_type, options, evidence, general_knowledge=general)
            system = build_system_prompt(self.settings.answer_grounding, self.settings.single_answer, self.settings.answer_context)
            if general:
                system += "\nGeneral-knowledge mode may be enabled below."
            try:
                with timer.stage("llm"):
                    raw = self.llm.generate(system, user_prompt, ANSWER_SCHEMA)
            except LLMError as exc:
                log.warning("llm failure: %s", exc)
                branch.status, branch.message = "error", f"Generarea raspunsului a esuat: {exc}"
                return branch
            with timer.stage("answer_validation"):
                try:
                    parsed = parse_answer(raw)
                except LLMError as exc:
                    branch.status, branch.message = "error", str(exc)
                    return branch
                self._fill_from_llm(branch, parsed, req, evidence, general, strict)
            return branch
        finally:
            branch.ms = int((time.perf_counter() - started) * 1000)
            branch.stages = dict(timer.stages)

    def _fill_from_llm(self, branch: Branch, parsed, req: AnswerRequest, evidence: list[Evidence], general: bool, strict: bool) -> None:
        allowed_labels = {o.label.upper() for o in req.options}
        allowed_ids = {e.id for e in evidence}
        cited = [i for i in parsed.evidence_ids if i in allowed_ids]
        branch.sources = self._sources_from_evidence(evidence, cited)
        if parsed.status != "answered":
            branch.message = parsed.explanation or "Dovezile disponibile nu sustin un raspuns."
            return
        labels = [l for l in parsed.option_labels if l in allowed_labels]
        if req.options and (not labels or len(labels) != len(parsed.option_labels)):
            branch.message = "Modelul a indicat optiuni care nu exista in intrebare."
            return
        if not req.options and not parsed.answer_text.strip():
            branch.sources, branch.message = [], "Modelul nu a produs un raspuns."
            return
        if strict and not cited and not general:
            branch.sources, branch.message = [], "Raspunsul nu a fost sustinut de nicio sursa citata; politica stricta de fundamentare l-a respins."
            return
        by_label = {o.label.upper(): o.text for o in req.options}
        text = "; ".join(by_label[l] for l in labels) if labels else parsed.answer_text.strip()
        explanation = parsed.explanation.strip()
        if not cited:
            explanation = (explanation + " (fara surse din documentele incarcate)").strip()
        branch.status, branch.labels, branch.text, branch.explanation = "answered", labels, text[:1000], explanation[:600]
        branch.general = not cited
        branch.multi = bool(req.options) and self.settings.single_answer and len(labels) > 1

    def _web_answer(self, req: AnswerRequest) -> Branch:
        branch, started = Branch(kind="web"), time.perf_counter()
        try:
            options = [(o.label.upper(), o.text) for o in req.options]
            result = self.web.answer(req.question, options, build_web_instructions(self.settings.answer_context, self.settings.single_answer))
        except LLMError as exc:
            log.warning("web search failure: %s", exc)
            branch.status, branch.message = "error", str(exc)
            return branch
        finally:
            branch.ms = int((time.perf_counter() - started) * 1000)
        branch.sources = [SourceRef(document_id="", document_name=(s.get("title") or domain_of(s["url"]))[:200], kind="web", url=s["url"])
                          for s in result.sources]
        if result.status != "answered":
            branch.status = "error" if result.status == "error" else "insufficient"
            branch.message = result.message or "Cautarea web nu a dat un raspuns."
            return branch
        allowed = {o.label.upper() for o in req.options}
        labels = [l for l in result.option_labels if l in allowed]
        if req.options and (not labels or len(labels) != len(result.option_labels)):
            branch.message = "Cautarea web a indicat optiuni care nu exista in intrebare."
            return branch
        if not req.options and not result.answer_text.strip():
            branch.message = "Cautarea web nu a produs un raspuns."
            return branch
        by_label = {o.label.upper(): o.text for o in req.options}
        text = "; ".join(by_label[l] for l in labels) if labels else result.answer_text.strip()
        branch.status, branch.labels, branch.text = "answered", labels, text[:1000]
        branch.explanation = result.explanation.strip()[:600]
        branch.multi = bool(req.options) and self.settings.single_answer and len(labels) > 1
        return branch

    def _decide_unknown(self, docs: Branch, web: Branch | None):
        """(status, answer, sources, message, match_type, agreement)."""
        d = docs if docs.usable else None
        w = web if web is not None and web.usable else None
        docs_type = "general_knowledge" if docs.general else "rag"
        if d is not None and w is not None:
            same = sorted(d.labels) == sorted(w.labels) if (d.labels or w.labels) else match_key(d.text) == match_key(w.text)
            if same:
                answer = Answer(option_labels=d.labels, text=d.text or w.text, explanation=(d.explanation or w.explanation)[:600])
                return ("answered", answer, d.sources + w.sources, "", "rag_web", "consensus")
            chosen, other = (w, d) if self.settings.web_search_prefer else (d, w)
            shown = ",".join(other.labels) or other.text[:60]
            explanation = f"Verifica: {_BRANCH_NAMES[other.kind]} indica {shown}. {chosen.explanation}".strip()[:600]
            message = f"Documentatia si internetul nu sunt de acord; este afisat raspunsul din {_BRANCH_NAMES[chosen.kind]}."
            answer = Answer(option_labels=chosen.labels, text=chosen.text, explanation=explanation)
            return ("answered", answer, chosen.sources + other.sources, message, "web" if chosen is w else docs_type, "disagreement")
        if w is not None:
            return ("answered", Answer(option_labels=w.labels, text=w.text, explanation=w.explanation), w.sources, "", "web", "web_only")
        if d is not None:
            return ("answered", Answer(option_labels=d.labels, text=d.text, explanation=d.explanation), d.sources, "", docs_type, "docs_only")
        if docs.status == "error" and (web is None or web.status in ("error", "timeout")):
            return ("error", None, [], docs.message, "none", "none")
        parts = []
        for branch in (docs, web):
            if branch is None:
                continue
            if branch.multi:
                parts.append(f"{_BRANCH_NAMES[branch.kind].capitalize()} a indicat mai multe optiuni ({','.join(branch.labels)}), "
                             "dar intrebarea are un singur raspuns corect.")
            elif branch.message:
                parts.append(branch.message)
        return ("insufficient_context", None, docs.sources, "; ".join(parts) or "Nicio sursa nu a dat un raspuns sigur.", "none", "none")

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
                sources: list[SourceRef] | None = None, message: str = "", match_type: str = "none",
                agreement: str = "none") -> AnswerResponse:
        with timer.stage("serialization"):
            response = AnswerResponse(request_id=req.request_id, status=status, answer=answer, sources=sources or [],
                                      message=message, match_type=match_type, agreement=agreement)
        response.timings_ms = dict(timer.stages)
        response.processing_time_ms = timer.total_ms
        return response
