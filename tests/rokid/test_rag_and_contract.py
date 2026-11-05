import pytest

from rokid_backend.llm.base import LLMError, extract_json, parse_answer
from helpers import admin_headers, ask, device_headers, example_guide_bytes, make_pdf_bytes, upload_document

OPTIONS = [("A", "List"), ("B", "Dictionary"), ("C", "Tuple"), ("D", "Set")]


def test_unseen_question_answered_from_documents(client, mock_llm):
    upload_document(client, "python_guide.md", example_guide_bytes())
    mock_llm.script({"status": "answered", "option_labels": ["C"], "answer_text": "Tuple", "explanation": "The guide says tuples are immutable.", "evidence_ids": ["S1"], "confidence": 0.9})
    r = ask(client, "Which of these Python containers cannot be changed after creation?", OPTIONS, request_id="rag-1")
    body = r.json()
    assert body["status"] == "answered" and body["match_type"] == "rag", body
    assert body["answer"]["option_labels"] == ["C"] and body["answer"]["text"] == "Tuple"
    assert body["sources"] and body["sources"][0]["document_name"] == "python_guide.md" and body["sources"][0]["kind"] == "document"
    assert body["timings_ms"]["llm"] >= 0 and body["timings_ms"]["vector_retrieval"] >= 0
    assert len(mock_llm.calls) == 1
    prompt = mock_llm.calls[0]["user"]
    assert "[S1] DOCUMENT" in prompt and "C. Tuple" in prompt and "python_guide.md" in prompt
    # the heuristic mock (no script) also finds the chunk that mentions tuples explicitly
    body = ask(client, "Which Python type is an immutable sequence?", OPTIONS).json()
    assert body["status"] in ("answered", "insufficient_context")


def test_pdf_sources_carry_page_numbers(client):
    upload_document(client, "guide.pdf", make_pdf_bytes(["Intro page about SQL.", "A tuple is an immutable sequence in Python and cannot be changed."]))
    body = ask(client, "Which Python container is immutable and cannot be changed?", OPTIONS).json()
    assert body["status"] == "answered" and body["sources"][0]["page"] == 2


def test_empty_knowledge_base_abstains_without_llm(client, mock_llm):
    body = ask(client, "Which Python type is immutable?", OPTIONS).json()
    assert body["status"] == "insufficient_context" and body["answer"] is None and body["sources"] == []
    assert mock_llm.calls == []


def test_llm_output_is_validated_against_request(client, mock_llm):
    upload_document(client, "python_guide.md", example_guide_bytes())
    mock_llm.script({"status": "answered", "option_labels": ["Z"], "answer_text": "x", "explanation": "", "evidence_ids": ["S1"], "confidence": 0.9})
    body = ask(client, "Which Python container type is immutable?", OPTIONS).json()
    assert body["status"] == "insufficient_context" and "nu exista" in body["message"]
    mock_llm.script({"status": "answered", "option_labels": ["C"], "answer_text": "Tuple", "explanation": "sure", "evidence_ids": [], "confidence": 0.9})
    body = ask(client, "Which Python container type is immutable?", OPTIONS).json()
    assert body["status"] == "insufficient_context" and "sursa" in body["message"]
    mock_llm.script({"status": "answered", "option_labels": ["C"], "answer_text": "Tuple", "explanation": "ok", "evidence_ids": ["S1", "S99"], "confidence": 0.9})
    body = ask(client, "Which Python container type is immutable?", OPTIONS).json()
    assert body["status"] == "answered" and len(body["sources"]) == 1
    mock_llm.script({"status": "insufficient_context", "option_labels": [], "answer_text": "", "explanation": "not covered", "evidence_ids": [], "confidence": 0.1})
    body = ask(client, "How many moons does Mars have in the Python guide?", OPTIONS).json()
    assert body["status"] == "insufficient_context" and body["message"] == "not covered"
    mock_llm.script({"status": "weird"})
    body = ask(client, "Which Python container type is immutable?", OPTIONS).json()
    assert body["status"] == "error"


def test_llm_failure_returns_error_status(client, mock_llm):
    upload_document(client, "python_guide.md", example_guide_bytes())

    def boom(system, user, schema):
        raise LLMError("timeout")
    mock_llm.generate = boom
    body = ask(client, "Which Python container type is immutable?", OPTIONS, request_id="e1").json()
    assert body["status"] == "error" and body["request_id"] == "e1" and body["answer"] is None


def test_general_knowledge_fallback_only_when_enabled(general_client, mock_llm):
    mock_llm.script({"status": "answered", "option_labels": ["C"], "answer_text": "Tuple", "explanation": "general", "evidence_ids": [], "confidence": 0.8})
    body = ask(general_client, "Which Python type is immutable?", OPTIONS).json()
    assert body["status"] == "answered" and body["match_type"] == "general_knowledge" and body["sources"] == []
    assert "fara surse" in body["answer"]["explanation"]


def test_open_question_rag(client, mock_llm):
    upload_document(client, "python_guide.md", example_guide_bytes())
    mock_llm.script({"status": "answered", "option_labels": [], "answer_text": "HAVING filters groups after GROUP BY.", "explanation": "see doc", "evidence_ids": ["S1"], "confidence": 0.8})
    body = ask(client, "Which SQL clause filters groups after aggregation?").json()
    assert body["status"] == "answered" and body["answer"]["option_labels"] == [] and "HAVING" in body["answer"]["text"]


def test_json_extraction_and_parsing():
    assert extract_json('```json\n{"status": "answered"}\n```')["status"] == "answered"
    assert extract_json('Sure! {"a": 1} done')["a"] == 1
    with pytest.raises(LLMError):
        extract_json("no json here")
    parsed = parse_answer({"status": "ANSWERED", "option_labels": ["c"], "evidence_ids": ["s1"], "confidence": "0.4"})
    assert parsed.option_labels == ["C"] and parsed.evidence_ids == ["S1"] and parsed.confidence == 0.4


def test_contract_shape_and_echo(client):
    body = ask(client, "Which Python type is immutable?", OPTIONS, request_id="uuid-123").json()
    assert set(body) >= {"request_id", "status", "answer", "sources", "processing_time_ms", "message"}
    assert body["request_id"] == "uuid-123" and isinstance(body["processing_time_ms"], int)


def test_invalid_questions(client):
    assert ask(client, "ab", OPTIONS).json()["status"] == "invalid_question"
    assert ask(client, "@@@ ### 12345 ***", OPTIONS).json()["status"] == "invalid_question"
    assert ask(client, "Which Python type is immutable?", [("A", "List"), ("B", "List")]).json()["status"] == "invalid_question"
    body = ask(client, "Which Python type is immutable?", OPTIONS, kb="nope").json()
    assert body["status"] == "invalid_question" and "nope" in body["message"]
    # neither text nor image
    r = client.post("/api/v1/questions/answer", data={"request_id": "x"}, headers=device_headers())
    assert r.status_code == 422 and "text" in r.json()["detail"]
    r = client.post("/api/v1/questions/answer", data={"text": "   "}, headers=device_headers())
    assert r.status_code == 422


def test_structured_validation_rules():
    from rokid_backend.rag.pipeline import AnswerService
    from rokid_backend.schemas.api import AnswerRequest, Option
    def req(options):
        return AnswerRequest(request_id="v", question="Which Python type is immutable?", options=[Option(label=l, text=t) for l, t in options])
    assert "singura optiune" in AnswerService._validate(req([("A", "List")]))
    assert "fara text" in AnswerService._validate(req([("A", "List"), ("B", "")]))
    assert "duplicate" in AnswerService._validate(req([("A", "List"), ("B", "List")]))
    assert AnswerService._validate(req(OPTIONS)) is None
    with pytest.raises(ValueError):
        req([("A", "1"), ("a", "2")])


def test_status_and_jobs_endpoints(client):
    upload_document(client, "python_guide.md", example_guide_bytes())
    status = client.get("/api/v1/admin/status", headers=admin_headers()).json()
    assert status["documents"] == {"indexed": 1} and status["llm_provider"] == "mock"
    jobs = client.get("/api/v1/admin/jobs", headers=admin_headers()).json()
    assert jobs[0]["kind"] == "document" and jobs[0]["status"] == "done"
    assert client.get("/admin").status_code == 200 and "Rokid" in client.get("/admin").text


def test_raw_block_is_split_into_question_and_options():
    from rokid_backend.core.rawtext import split_raw_question
    q, opts = split_raw_question("Tuturor navelor aflate în largul mării li se aplică prevederile:\nA  COLREG-1972.\nB  R.N.D.\nC  Codului ISPS.")
    assert q.startswith("Tuturor navelor") and opts == [("A", "COLREG-1972."), ("B", "R.N.D."), ("C", "Codului ISPS.")]
    q, opts = split_raw_question("Which type is immutable? A. List B. Tuple C. Set")
    assert q == "Which type is immutable?" and opts == [("A", "List"), ("B", "Tuple"), ("C", "Set")]
    q, opts = split_raw_question("A fost odata ca niciodata?\n1) da\n2) nu, cu o continuare\npe doua linii")
    assert q == "A fost odata ca niciodata?" and opts == [("A", "da"), ("B", "nu, cu o continuare pe doua linii")]
    assert split_raw_question("Just a question without options") == ("Just a question without options", [])


def test_raw_text_request_answers_from_verified_record(client):
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    body = ask(client, "Which Python type is immutable?\nA. Set\nB. Tuple\nC. List\nD. Dictionary").json()
    assert body["status"] == "answered" and body["answer"]["option_labels"] == ["B"] and body["match_type"] == "exact_qa"
    body = ask(client, "", raw_ocr_text="Which Python type is immutable?\nA. Set\nB. Tuple\nC. List").json()
    assert body["status"] == "answered" and body["answer"]["option_labels"] == ["B"]


def _png() -> bytes:
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 20), False)
    pix.clear_with(255)
    return pix.tobytes("png")


def test_image_only_request(client, mock_llm):
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    mock_llm.transcriptions.append("Which Python type is immutable?\nA. Set\nB. Tuple\nC. List")
    r = client.post("/api/v1/questions/answer?full=1", files={"image": ("q.png", _png(), "image/png")}, data={"request_id": "img-7"}, headers=device_headers())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "img-7" and body["status"] == "answered" and body["answer"]["option_labels"] == ["B"]
    assert body["transcribed_text"].startswith("Which Python type") and "transcription" in body["timings_ms"]
    assert mock_llm.images[0][1] == "image/png"
    mock_llm.transcriptions.append("Which Python type is immutable?\nA. Set\nB. Tuple\nC. List")
    r = client.post("/api/v1/questions/answer", files={"image": ("q.png", _png(), "image/png")}, headers=device_headers())
    assert r.json() == {"label": "B", "text": "Tuple"}
    # not an image
    r = client.post("/api/v1/questions/answer", files={"image": ("q.txt", b"hello", "text/plain")}, headers=device_headers())
    assert r.status_code == 200 and "imagine" in r.json()["error"]
    # provider cannot read images and no OCR -> clear error, no label
    r = client.post("/api/v1/questions/answer", files={"image": ("q.png", _png(), "image/png")}, headers=device_headers())
    assert "Nu am putut extrage" in r.json()["error"] and "label" not in r.json()
    mock_llm.transcriptions.append("NO_QUESTION")
    r = client.post("/api/v1/questions/answer", files={"image": ("q.png", _png(), "image/png")}, headers=device_headers())
    assert "intrebare" in r.json()["error"]
    assert client.post("/api/v1/questions/answer", files={"image": ("q.png", _png(), "image/png")}).status_code == 401


def test_text_first_image_as_fallback(client, mock_llm):
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    good = "Which Python type is immutable?\nA. Set\nB. Tuple\nC. List"
    # text answers -> the image is never sent to the vision model
    mock_llm.transcriptions.append(good)
    r = client.post("/api/v1/questions/answer?full=1", data={"text": good}, files={"image": ("q.png", _png(), "image/png")}, headers=device_headers())
    body = r.json()
    assert body["status"] == "answered" and body["transcribed_text"] is None and mock_llm.images == []
    # garbled OCR text from the glasses -> the screenshot is transcribed and answered
    r = client.post("/api/v1/questions/answer?full=1", data={"text": "Wh1ch Pyth"}, files={"image": ("q.png", _png(), "image/png")}, headers=device_headers())
    body = r.json()
    assert body["status"] == "answered" and body["answer"]["option_labels"] == ["B"] and body["transcribed_text"] == good and len(mock_llm.images) == 1
    r = client.post("/api/v1/questions/answer", data={"text": "Wh1ch Pyth"}, files={"image": ("q.png", _png(), "image/png")}, headers=device_headers())
    # second call: no scripted transcription -> the text error is kept and the image problem is appended
    assert "label" not in r.json() and "Poza:" in r.json()["error"]


def test_compact_default_reply(client):
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    r = client.post("/api/v1/questions/answer", data={"request_id": "c1", "text": "Which Python type is immutable?\nA. Set\nB. Tuple"}, headers=device_headers())
    assert r.json() == {"label": "B", "text": "Tuple"}
    block = "Which of the following are valid ways to create an empty dictionary?\nA. set()\nB. {}\nC. dict()"
    assert client.post("/api/v1/questions/answer", data={"text": block}, headers=device_headers()).json() == {"label": "B,C", "text": "{}; dict()"}
    r = client.post("/api/v1/questions/answer", data={"request_id": "c2", "text": "ab"}, headers=device_headers())
    assert list(r.json()) == ["error"]
    r = client.post("/api/v1/questions/answer", data={"request_id": "c3", "text": "What does len('abc') return?"}, headers=device_headers())
    assert r.json() == {"text": "3"}
    full = client.post("/api/v1/questions/answer?full=1", data={"request_id": "c4", "text": "Which Python type is immutable?\nA. Set\nB. Tuple"}, headers=device_headers()).json()
    assert full["status"] == "answered" and full["sources"] and full["request_id"] == "c4"
    # urlencoded form works too (no multipart needed for text-only calls)
    r = client.post("/api/v1/questions/answer", content="text=Which+Python+type+is+immutable%3F%0AA.+Set%0AB.+Tuple",
                    headers={**device_headers(), "Content-Type": "application/x-www-form-urlencoded"})
    assert r.json() == {"label": "B", "text": "Tuple"}


ROKID_BODY = {
    "request_id": "7a5c6b9e-1b8e-4d1a-9d2c-4d6f0c9a1e33", "knowledge_base_id": "default", "source_mode": "rokid_ocr",
    "question": "Which Python type is immutable?", "question_type": "multiple_choice",
    "options": [{"label": "A", "text": "List"}, {"label": "B", "text": "Dictionary"}, {"label": "C", "text": "Tuple"}, {"label": "D", "text": "Set"}],
    "raw_ocr_text": "Which Python type is immutable?\nA. List\nB. Dictionary\nC. Tuple\nD. Set", "client_confidence": 0.93,
}


def test_rokid_json_contract(client, mock_llm):
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    r = client.post("/api/v1/questions/answer", json=ROKID_BODY, headers={**device_headers(), "X-Request-Id": ROKID_BODY["request_id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == ROKID_BODY["request_id"] and body["status"] == "answered"
    assert body["answer"]["option_labels"] == ["C"] and body["answer"]["text"] == "Tuple" and body["answer"]["explanation"]
    assert body["sources"][0]["document_name"] and isinstance(body["processing_time_ms"], int) and "message" in body
    assert r.headers["X-Request-ID"] == ROKID_BODY["request_id"]
    # the JSON shape never gets the compact reply, with or without ?full
    assert set(client.post("/api/v1/questions/answer?full=0", json=ROKID_BODY, headers=device_headers()).json()) >= {"status", "answer", "sources"}
    # client aliases for question_type and open questions
    r = client.post("/api/v1/questions/answer", json={"request_id": "ms-1", "question": "Which of the following are valid ways to create an empty dictionary?",
                                                      "question_type": "multiple_select", "options": [{"label": "A", "text": "set()"}, {"label": "B", "text": "{}"}, {"label": "C", "text": "dict()"}]},
                    headers=device_headers())
    assert sorted(r.json()["answer"]["option_labels"]) == ["B", "C"]
    r = client.post("/api/v1/questions/answer", json={"request_id": "oe-1", "question": "What does len('abc') return?", "question_type": "open_ended", "options": []}, headers=device_headers())
    assert r.json()["status"] == "answered" and r.json()["answer"]["text"] == "3" and r.json()["answer"]["option_labels"] == []
    # business errors are HTTP 200 with a status
    r = client.post("/api/v1/questions/answer", json={"request_id": "bad-1", "question": "ab", "options": []}, headers=device_headers())
    assert r.status_code == 200 and r.json()["status"] == "invalid_question" and r.json()["answer"] is None
    r = client.post("/api/v1/questions/answer", json={"request_id": "bad-2", "question": "q?", "options": [{"label": "A", "text": "1"}, {"label": "a", "text": "2"}]}, headers=device_headers())
    assert r.status_code == 200 and r.json()["status"] == "invalid_question" and "options" in r.json()["message"]
    r = client.post("/api/v1/questions/answer", json={"request_id": "bad-3"}, headers=device_headers())
    assert r.status_code == 200 and r.json()["status"] == "invalid_question"
    # unreadable bodies are the only 422s
    assert client.post("/api/v1/questions/answer", json={"question": "no id"}, headers=device_headers()).status_code == 422
    assert client.post("/api/v1/questions/answer", content=b"{not json", headers={**device_headers(), "Content-Type": "application/json"}).status_code == 422
    assert client.post("/api/v1/questions/answer", json=[1, 2], headers=device_headers()).status_code == 422
    # health keep-alive with the device bearer
    assert client.get("/api/v1/health", headers=device_headers()).json()["status"] == "ok"


def test_rokid_inline_image_fallback(client, mock_llm):
    import base64
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    good = "Which Python type is immutable?\nA. Set\nB. Tuple\nC. List"
    image = {"format": "png", "base64": base64.b64encode(_png()).decode(), "width": 40, "height": 20}
    # confident OCR text answers -> the image is not transcribed
    r = client.post("/api/v1/questions/answer", json={"request_id": "im-1", "question": good, "options": [], "image": image}, headers=device_headers())
    assert r.json()["status"] == "answered" and r.json()["transcribed_text"] is None and mock_llm.images == []
    # garbled OCR -> image transcribed and answered
    mock_llm.transcriptions.append(good)
    r = client.post("/api/v1/questions/answer", json={"request_id": "im-2", "question": "Wh1ch Pyth", "options": [], "image": image}, headers=device_headers())
    body = r.json()
    assert body["status"] == "answered" and body["answer"]["option_labels"] == ["B"] and body["transcribed_text"] == good and len(mock_llm.images) == 1
    # data: URL prefix accepted; invalid base64 is a clean business error
    mock_llm.transcriptions.append(good)
    r = client.post("/api/v1/questions/answer", json={"request_id": "im-3", "image": {"base64": "data:image/png;base64," + image["base64"]}}, headers=device_headers())
    assert r.json()["status"] == "answered"
    r = client.post("/api/v1/questions/answer", json={"request_id": "im-4", "image": {"base64": "not*base64"}}, headers=device_headers())
    assert r.json()["status"] == "invalid_question" and "imagine" in r.json()["message"]


def test_idempotent_replies_per_request_id(client, mock_llm):
    from helpers import example_dataset_bytes, upload_dataset
    upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
    first = client.post("/api/v1/questions/answer", json=ROKID_BODY, headers=device_headers()).json()
    # a retry with the same id (even with a different body) gets the stored reply, byte for byte
    changed = {**ROKID_BODY, "raw_ocr_text": None, "options": [{"label": "A", "text": "Set"}, {"label": "B", "text": "Tuple"}, {"label": "C", "text": "List"}]}
    again = client.post("/api/v1/questions/answer", json=changed, headers=device_headers()).json()
    assert again == first and again["answer"]["option_labels"] == ["C"]
    # a different id is computed afresh
    fresh = client.post("/api/v1/questions/answer", json={**changed, "request_id": "other-1"}, headers=device_headers()).json()
    assert fresh["answer"]["option_labels"] == ["B"]
    # the form shape is idempotent too
    a = client.post("/api/v1/questions/answer?full=1", data={"request_id": "f-1", "text": "Which Python type is immutable?\nA. Set\nB. Tuple"}, headers=device_headers()).json()
    b = client.post("/api/v1/questions/answer?full=1", data={"request_id": "f-1", "text": "Which Python type is immutable?\nA. Tuple\nB. Set"}, headers=device_headers()).json()
    assert a == b and a["answer"]["option_labels"] == ["B"]
    # errors are also replayed; a new id after TTL 0 would recompute (cache disabled)
    from rokid_backend.core.idempotency import IdempotencyCache
    off = IdempotencyCache(ttl_seconds=0)
    off.put("k", "v")
    assert off.get("k") is None
