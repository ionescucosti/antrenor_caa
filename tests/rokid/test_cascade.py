"""Unknown questions: documentation (+ model knowledge) and web search in
parallel, consensus / disagreement decisions, candidate review in admin,
the web-search provider request shape, and client retries."""
from __future__ import annotations

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from helpers import admin_headers, ask, device_headers, example_guide_bytes, make_settings, upload_document
from rokid_backend.llm.base import LLMError
from rokid_backend.llm.openai_provider import OpenAIProvider
from rokid_backend.llm.web_search import OpenAIWebSearch, WebAnswer, domain_of
from rokid_backend.main import create_app
from rokid_backend.rag.embeddings import EmbeddingError, OpenAIEmbedding
from rokid_backend.rag.prompts import build_system_prompt, build_web_instructions, build_web_input

OPTIONS = [("A", "List"), ("B", "Dictionary"), ("C", "Tuple"), ("D", "Set")]
QUESTION = "Which of these Python containers cannot be changed after creation?"


class FakeWeb:
    name = "fake-web"

    def __init__(self) -> None:
        self.queue: list[WebAnswer] = []
        self.calls: list[tuple] = []
        self.delay = 0.0

    def push(self, **kwargs) -> None:
        self.queue.append(WebAnswer(**kwargs))

    def answer(self, question, options, instructions):
        self.calls.append((question, options, instructions))
        time.sleep(self.delay)
        if not self.queue:
            raise LLMError("fara web")
        item = self.queue.pop(0)
        if item.status == "raise":
            raise LLMError(item.message)
        return item


def answered(labels, explanation="web says", **extra):
    return dict(status="answered", option_labels=list(labels), answer_text="", explanation=explanation, confidence=0.9,
                sources=[{"url": "https://docs.python.org/3/tutorial/datastructures.html", "title": "Data Structures"}], **extra)


@pytest.fixture
def web():
    return FakeWeb()


@pytest.fixture
def open_client(tmp_path, mock_llm, web):
    settings = make_settings(tmp_path, answer_grounding="open", web_search_enabled=True, web_search_timeout_seconds=1.0, docs_timeout_seconds=5.0)
    app = create_app(settings, llm=mock_llm, web=web)
    with TestClient(app) as c:
        c.app = app
        yield c


def _docs_answer(mock_llm, labels):
    mock_llm.script({"status": "answered", "option_labels": labels, "answer_text": "", "explanation": "the guide says so",
                     "evidence_ids": ["S1"], "confidence": 0.9})


# --- prompts ----------------------------------------------------------------

def test_prompts_open_strict_and_web():
    open_prompt = build_system_prompt("open", single_answer=True, context="ANR exam context.")
    assert "ANR exam context." in open_prompt and "exactly ONE correct option" in open_prompt and "own expert knowledge" in open_prompt
    assert "Do not use outside knowledge" not in open_prompt
    strict = build_system_prompt("strict", single_answer=False)
    assert "Do not use outside knowledge" in strict and "Several labels are allowed" in strict and "exactly ONE" not in strict
    web = build_web_instructions("ANR exam context.", True)
    assert "ANR exam context." in web and "exactly one correct option" in web and "JSON" in web
    assert build_web_input(" Q ? ", [("A", "x"), ("B", "y")]) == "QUESTION: Q ?\nOPTIONS:\nA. x\nB. y"


# --- decisions ----------------------------------------------------------------

def test_consensus_between_documentation_and_web(open_client, mock_llm, web):
    upload_document(open_client, "python_guide.md", example_guide_bytes())
    _docs_answer(mock_llm, ["C"])
    web.push(**answered("C"))
    body = ask(open_client, QUESTION, OPTIONS, request_id="c-1").json()
    assert body["status"] == "answered" and body["match_type"] == "rag_web" and body["agreement"] == "consensus", body
    assert body["answer"]["option_labels"] == ["C"] and body["answer"]["text"] == "Tuple" and body["message"] == ""
    kinds = {s["kind"] for s in body["sources"]}
    assert kinds == {"document", "web"}
    web_source = next(s for s in body["sources"] if s["kind"] == "web")
    assert web_source["url"].startswith("https://docs.python.org") and web_source["document_name"] == "Data Structures"
    assert body["candidate_id"] and body["timings_ms"]["web_ms"] >= 0 and body["timings_ms"]["docs_ms"] >= 0 and body["timings_ms"]["llm"] >= 0
    question, options, instructions = web.calls[0]
    assert question == QUESTION and options == [("A", "List"), ("B", "Dictionary"), ("C", "Tuple"), ("D", "Set")] and "exactly one" in instructions
    # the open system prompt reached the LLM
    assert "own expert knowledge" in mock_llm.calls[0]["system"]


def test_disagreement_shows_web_answer_marked(open_client, mock_llm, web):
    upload_document(open_client, "python_guide.md", example_guide_bytes())
    _docs_answer(mock_llm, ["B"])
    web.push(**answered("C", explanation="Tuples are immutable (docs.python.org)."))
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "answered" and body["match_type"] == "web" and body["agreement"] == "disagreement"
    assert body["answer"]["option_labels"] == ["C"]
    assert body["answer"]["explanation"].startswith("Verifica: documentatia indica B.") and "nu sunt de acord" in body["message"]
    # the compact reply for the glasses carries a note
    _docs_answer(mock_llm, ["B"])
    web.push(**answered("C"))
    text = "\n".join([QUESTION] + [f"{l}. {t}" for l, t in OPTIONS])
    compact = open_client.post("/api/v1/questions/answer", data={"text": text}, headers=device_headers()).json()
    assert compact == {"label": "C", "text": "Tuple", "note": "verifica: documentatia si internetul nu sunt de acord"}


def test_web_only_and_docs_only(open_client, mock_llm, web):
    upload_document(open_client, "python_guide.md", example_guide_bytes())
    mock_llm.script({"status": "insufficient_context", "option_labels": [], "answer_text": "", "explanation": "not covered", "evidence_ids": [], "confidence": 0.1})
    web.push(**answered("C"))
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "answered" and body["match_type"] == "web" and body["agreement"] == "web_only"
    assert all(s["kind"] == "web" for s in body["sources"])
    # web fails -> documentation alone
    _docs_answer(mock_llm, ["C"])
    web.push(status="raise", message="HTTP 500")
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "answered" and body["match_type"] == "rag" and body["agreement"] == "docs_only" and body["message"] == ""
    # web too slow -> documentation alone, within the deadline
    _docs_answer(mock_llm, ["C"])
    web.delay = 2.5
    web.push(**answered("B"))
    started = time.perf_counter()
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "answered" and body["agreement"] == "docs_only" and body["answer"]["option_labels"] == ["C"]
    assert time.perf_counter() - started < 2.4


def test_multi_label_answer_is_unsure(open_client, mock_llm, web):
    upload_document(open_client, "python_guide.md", example_guide_bytes())
    _docs_answer(mock_llm, ["B", "C"])
    web.push(status="insufficient_context", message="nothing found")
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "insufficient_context" and "mai multe optiuni" in body["message"] and "nothing found" in body["message"]
    _docs_answer(mock_llm, ["B", "C"])
    web.push(**answered("C"))
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "answered" and body["agreement"] == "web_only" and body["answer"]["option_labels"] == ["C"]


def test_open_grounding_answers_without_evidence(open_client, mock_llm, web):
    mock_llm.script({"status": "answered", "option_labels": ["C"], "answer_text": "", "explanation": "known fact", "evidence_ids": [], "confidence": 0.5})
    body = ask(open_client, QUESTION, OPTIONS).json()   # empty knowledge base, web raises
    assert body["status"] == "answered" and body["match_type"] == "general_knowledge" and body["agreement"] == "docs_only"
    assert "fara surse" in body["answer"]["explanation"] and body["sources"] == []
    assert len(mock_llm.calls) == 1
    # both branches fail -> error status with the documentation message
    def boom(system, user, schema):
        raise LLMError("timeout")
    mock_llm.generate = boom
    body = ask(open_client, QUESTION, OPTIONS).json()
    assert body["status"] == "error" and "timeout" in body["message"]


# --- candidates -----------------------------------------------------------------

def test_candidate_confirm_makes_the_question_verified(open_client, mock_llm, web):
    upload_document(open_client, "python_guide.md", example_guide_bytes())
    for _ in range(2):   # two frames of the same screen: one pending candidate, seen twice
        _docs_answer(mock_llm, ["C"])
        web.push(**answered("C"))
        body = ask(open_client, QUESTION, OPTIONS).json()
        assert body["agreement"] == "consensus"
    pending = open_client.get("/api/v1/admin/candidates?status=pending", headers=admin_headers()).json()
    assert len(pending) == 1 and pending[0]["seen_count"] == 2 and pending[0]["option_labels"] == ["C"]
    assert pending[0]["docs_labels"] == ["C"] and pending[0]["web_labels"] == ["C"] and pending[0]["agreement"] == "consensus"
    status = open_client.get("/api/v1/admin/status", headers=admin_headers()).json()
    assert status["candidates_pending"] == 1 and status["web_search"] == "fake-web" and status["answer_grounding"] == "open"
    # confirm with a corrected label
    r = open_client.post(f"/api/v1/admin/candidates/{pending[0]['id']}/confirm", json={"option_labels": ["B"]}, headers=admin_headers())
    assert r.status_code == 200 and r.json()["record_id"].startswith("rec_")
    record_id = r.json()["record_id"]
    body = ask(open_client, QUESTION, [("A", "Tuple"), ("B", "List"), ("C", "Dictionary"), ("D", "Set")]).json()
    assert body["status"] == "answered" and body["match_type"] == "exact_qa" and body["agreement"] == "verified"
    assert body["answer"]["option_labels"] == ["C"] and body["answer"]["text"] == "Dictionary"   # mapped by text, not letter
    assert mock_llm.calls and len(web.calls) == 2   # no new LLM / web calls for a verified question
    sets = open_client.get("/api/v1/qa-datasets", headers=admin_headers()).json()
    confirmed = next(s for s in sets if s["format"] == "confirmed")
    assert confirmed["name"] == "Confirmate din admin" and confirmed["record_count"] == 1 and confirmed["status"] == "indexed"
    records = open_client.get(f"/api/v1/qa-datasets/{confirmed['id']}/records", headers=admin_headers()).json()
    assert records["total"] == 1 and records["records"][0]["correct_labels"] == ["B"] and records["records"][0]["source"]["kind"] == "confirmed"
    # the confirmed set can be reindexed without a source file
    r = open_client.post(f"/api/v1/qa-datasets/{confirmed['id']}/reindex", headers=admin_headers())
    assert r.status_code == 202 and open_client.app.state.store.count("questions", "default") == 1
    confirmed_list = open_client.get("/api/v1/admin/candidates?status=confirmed", headers=admin_headers()).json()
    assert len(confirmed_list) == 1 and confirmed_list[0]["record_id"] == record_id and confirmed_list[0]["option_labels"] == ["B"]
    # reject another candidate; bad requests are reported
    _docs_answer(mock_llm, ["A"])
    web.push(**answered("A"))
    other = ask(open_client, "Which Python container keeps insertion order and allows duplicates?", OPTIONS).json()
    assert other["candidate_id"]
    assert open_client.post(f"/api/v1/admin/candidates/{other['candidate_id']}/reject", headers=admin_headers()).status_code == 200
    assert open_client.get("/api/v1/admin/candidates?status=rejected", headers=admin_headers()).json()[0]["id"] == other["candidate_id"]
    assert open_client.post(f"/api/v1/admin/candidates/{other['candidate_id']}/confirm", json={"option_labels": ["Z"]}, headers=admin_headers()).status_code == 400
    assert open_client.post("/api/v1/admin/candidates/cand_missing/confirm", headers=admin_headers()).status_code == 404
    assert open_client.get("/api/v1/admin/candidates", headers=device_headers()).status_code == 403


# --- web search provider ----------------------------------------------------------

def _responses_payload(text, cited=("https://legislatie.just.ro/Public/DetaliiDocument/1",), consulted=("https://example.org/a", "https://legislatie.just.ro/Public/DetaliiDocument/1")):
    return {"status": "completed", "usage": {"input_tokens": 9000, "output_tokens": 100}, "output": [
        {"type": "web_search_call", "id": "ws_1", "status": "completed",
         "action": {"type": "search", "query": "colreg", "sources": [{"type": "url", "url": u} for u in consulted]}},
        {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [
            {"type": "output_text", "text": text, "annotations": [{"type": "url_citation", "url": u, "title": "Lege", "start_index": 0, "end_index": 1} for u in cited]}]}]}


def _provider(handler, **kwargs) -> OpenAIWebSearch:
    provider = OpenAIWebSearch("sk-test", "gpt-6.1-terra", blocked_domains="forum.example, https://www.scribd.com/", **kwargs)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    return provider


def test_web_search_request_and_response_shape():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json=_responses_payload(json.dumps({"status": "answered", "option_labels": ["c"], "answer_text": "Tuple",
                                                                       "explanation": "Legea spune.", "evidence_ids": [], "confidence": 0.95})))
    result = _provider(handler).answer("Q?", [("A", "x"), ("B", "y"), ("C", "z")], "INSTR")
    body = seen[0]
    tool = body["tools"][0]
    assert tool["type"] == "web_search" and tool["search_context_size"] == "medium" and tool["user_location"] == {"type": "approximate", "country": "RO"}
    assert tool["filters"] == {"blocked_domains": ["forum.example", "scribd.com"]}
    assert body["tool_choice"] == "required" and body["include"] == ["web_search_call.action.sources"] and body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
    assert body["instructions"] == "INSTR" and body["input"] == "QUESTION: Q?\nOPTIONS:\nA. x\nB. y\nC. z" and body["model"] == "gpt-6.1-terra"
    assert result.status == "answered" and result.option_labels == ["C"] and result.answer_text == "Tuple" and result.confidence == 0.95
    assert [s["url"] for s in result.sources] == ["https://legislatie.just.ro/Public/DetaliiDocument/1", "https://example.org/a"]
    assert result.sources[0]["title"] == "Lege" and result.searches == 1 and result.ms >= 0
    assert domain_of("https://www.legislatie.just.ro/x") == "legislatie.just.ro"


def test_web_search_adapts_parameters_and_reports_errors():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "text" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: text.format", "param": "text.format", "type": "invalid_request_error"}})
        if len(calls) == 2:
            return httpx.Response(200, json=_responses_payload('Here you go:\n```json\n{"status": "answered", "option_labels": ["B"], "answer_text": "y", "explanation": "", "evidence_ids": [], "confidence": 0.8}\n```'))
        if len(calls) == 3:
            return httpx.Response(200, json=_responses_payload('{"status": "insufficient_context", "option_labels": [], "answer_text": "", "explanation": "nu am gasit", "evidence_ids": [], "confidence": 0.2}'))
        if len(calls) == 4:
            return httpx.Response(200, json={"status": "incomplete", "output": []})
        return httpx.Response(503, json={"error": {"message": "overloaded"}})
    provider = _provider(handler, retries=0)
    result = provider.answer("Q?", [("A", "x"), ("B", "y")], "I")
    assert result.status == "answered" and result.option_labels == ["B"] and "text" not in calls[1] and not provider._structured
    assert provider.answer("Q?", [("A", "x"), ("B", "y")], "I").status == "insufficient_context"
    broken = provider.answer("Q?", [("A", "x"), ("B", "y")], "I")
    assert broken.status == "error" and "incomplete" in broken.message
    with pytest.raises(LLMError, match="HTTP 503"):
        provider.answer("Q?", [("A", "x"), ("B", "y")], "I")

    def network(request):
        raise httpx.ConnectError("down")
    with pytest.raises(LLMError, match="ConnectError"):
        _provider(network).answer("Q?", [], "I")


# --- retries ---------------------------------------------------------------------

def test_embeddings_and_llm_clients_retry_transient_failures():
    attempts = {"emb": 0, "llm": 0}

    def embeddings(request: httpx.Request) -> httpx.Response:
        attempts["emb"] += 1
        if attempts["emb"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limit"}})
        if attempts["emb"] == 2:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
    emb = OpenAIEmbedding("sk-test", "text-embedding-3-small", "https://api.openai.com/v1", retries=2)
    emb._client = httpx.Client(transport=httpx.MockTransport(embeddings), base_url="https://api.openai.com/v1")
    assert emb.embed(["x"]) == [[0.6, 0.8]] and attempts["emb"] == 3

    def chat(request: httpx.Request) -> httpx.Response:
        attempts["llm"] += 1
        if attempts["llm"] == 1:
            raise httpx.ConnectError("reset")
        if attempts["llm"] == 2:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"status": "answered"}'}, "finish_reason": "stop"}]})
    llm = OpenAIProvider("sk-test", "gpt-6.1-terra", retries=2)
    llm._client = httpx.Client(transport=httpx.MockTransport(chat), base_url="https://api.openai.com/v1")
    assert llm.generate("s", "u", {"type": "object"}) == {"status": "answered"} and attempts["llm"] == 3

    def always_down(request):
        raise httpx.ReadTimeout("slow")
    llm = OpenAIProvider("sk-test", "gpt-6.1-terra", retries=1)
    llm._client = httpx.Client(transport=httpx.MockTransport(always_down), base_url="https://api.openai.com/v1")
    with pytest.raises(LLMError, match="ReadTimeout"):
        llm.generate("s", "u", {"type": "object"})


def test_exhausted_quota_is_not_retried():
    attempts = {"n": 0}
    quota = {"error": {"message": "You have no credits remaining.", "type": "insufficient_quota", "code": "credit_balance_exhausted"}}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, json=quota)
    emb = OpenAIEmbedding("sk-test", "text-embedding-3-small", "https://api.openai.com/v1", retries=2)
    emb._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    with pytest.raises(EmbeddingError, match="credit"):
        emb.embed(["x"])
    llm = OpenAIProvider("sk-test", "gpt-6.1-terra", retries=2)
    llm._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    with pytest.raises(LLMError, match="credit"):
        llm.generate("s", "u", {"type": "object"})
    web = OpenAIWebSearch("sk-test", "gpt-6.1-terra", retries=2)
    web._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.openai.com/v1")
    with pytest.raises(LLMError, match="credit"):
        web.answer("Q?", [], "I")
    assert attempts["n"] == 3   # one call each, no retries


def test_embedding_failure_is_reported_not_crashed(tmp_path, mock_llm):
    class BrokenEmbedder:
        name, dimension = "broken", 3

        def embed(self, texts):
            raise EmbeddingError("Contul OpenAI nu mai are credit (insufficient_quota).")
    app = create_app(make_settings(tmp_path), llm=mock_llm, embedder=BrokenEmbedder())
    with TestClient(app) as client:
        body = ask(client, QUESTION, OPTIONS).json()
        assert body["status"] == "error" and "credit" in body["message"]
