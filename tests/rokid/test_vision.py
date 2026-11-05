"""Photo path: image preparation, transcription prompt and cleanup, the OpenAI
request shape across model generations, Chroma after an embedding change, and
the admin status fields that describe them."""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from helpers import admin_headers, example_dataset_bytes, make_settings, upload_dataset
from rokid_backend.core.config import DEFAULT_TRANSCRIBE_CONTEXT, Settings
from rokid_backend.llm.base import LLMError
from rokid_backend.llm.mock import MockLLM
from rokid_backend.llm.openai_provider import OpenAIProvider, is_reasoning_model
from rokid_backend.main import create_app
from rokid_backend.models import QADataset
from rokid_backend.rag.imageprep import exif_orientation, prepare_image
from rokid_backend.rag.prompts import build_transcribe_prompt
from rokid_backend.rag.transcribe import clean_transcription, transcribe_image
from rokid_backend.rag.vectorstore import VectorStore


def _jpeg(width: int, height: int) -> bytes:
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), 0)
    pix.clear_with(255)
    return pix.tobytes("jpeg", jpg_quality=85)


def _png(width: int = 40, height: int = 20) -> bytes:
    import fitz
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), 0)
    pix.clear_with(200)
    return pix.tobytes("png")


def _with_exif_orientation(jpeg: bytes, orientation: int) -> bytes:
    tiff = b"MM\x00\x2a" + (8).to_bytes(4, "big")
    entry = (0x0112).to_bytes(2, "big") + (3).to_bytes(2, "big") + (1).to_bytes(4, "big") + orientation.to_bytes(2, "big") + b"\x00\x00"
    payload = b"Exif\x00\x00" + tiff + (1).to_bytes(2, "big") + entry + (0).to_bytes(4, "big")
    return jpeg[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + jpeg[2:]


# --- image preparation --------------------------------------------------------

def test_prepare_image_downscales_large_photos_and_keeps_small_ones():
    import fitz
    big = _jpeg(4000, 3000)
    out, mime, info = prepare_image(big, "image/jpeg", 1792)
    assert mime == "image/jpeg" and info["resized"] and (info["width"], info["height"]) == (1792, 1344)
    assert len(out) < len(big) and (fitz.Pixmap(out).width, fitz.Pixmap(out).height) == (1792, 1344)
    small = _png()
    assert prepare_image(small, "image/png", 1792) == (small, "image/png", {"original_bytes": len(small), "resized": False, "width": 40, "height": 20})
    assert prepare_image(big, "image/jpeg", 0)[0] == big   # 0 = never resize


def test_prepare_image_leaves_rotated_jpegs_and_junk_alone():
    rotated = _with_exif_orientation(_jpeg(3000, 2000), 6)
    assert exif_orientation(rotated) == 6 and exif_orientation(_jpeg(10, 10)) == 1 and exif_orientation(_png()) == 1
    out, mime, info = prepare_image(rotated, "image/jpeg", 1792)
    assert out == rotated and mime == "image/jpeg" and info["skipped"] == "exif-orientation"
    junk = b"\x89PNG\r\n\x1a\n" + b"not really a png"
    out, mime, info = prepare_image(junk, "image/png", 1792)
    assert out == junk and mime == "image/png" and "skipped" in info


# --- prompt and cleanup --------------------------------------------------------

def test_transcribe_prompt_carries_the_domain_context_and_settings_defaults():
    prompt = build_transcribe_prompt(DEFAULT_TRANSCRIBE_CONTEXT)
    assert "NO_QUESTION" in prompt and "ANR" in prompt and "A. <text of the first option>" in prompt
    assert "ANR" not in build_transcribe_prompt("") and "NO_QUESTION" in build_transcribe_prompt("")
    settings = Settings(_env_file=None)
    assert settings.vision_reasoning_effort == "none" and settings.llm_reasoning_effort == "low"
    assert settings.vision_detail == "high" and settings.vision_max_side == 1792 and settings.transcribe_context == DEFAULT_TRANSCRIBE_CONTEXT
    assert Settings(_env_file=None, llm_provider="openai", openai_model="gpt-6.1-terra", vision_model="").vision_model_name == "gpt-6.1-terra"
    assert Settings(_env_file=None, llm_provider="openai", openai_model="gpt-6.1-terra", vision_model="gpt-6-luna").vision_model_name == "gpt-6-luna"


def test_clean_transcription_normalises_model_output():
    raw = ("```text\n**Întrebare 12:** 12. Căror nave li se aplică prevederile Colreg-1972?\nOpțiuni:\n"
           "A. tuturor navelor care navigă în apele internaționale.\nB. tuturor navelor care navigă în rade.\n"
           "C. tuturor navelor aflate în largul mării.\n```")
    assert clean_transcription(raw) == ("Căror nave li se aplică prevederile Colreg-1972?\nA. tuturor navelor care navigă în apele internaționale.\n"
                                        "B. tuturor navelor care navigă în rade.\nC. tuturor navelor aflate în largul mării.")
    assert clean_transcription("Question:\n\nWhich Python type is immutable?\nA. Set\nB. Tuple") == "Which Python type is immutable?\nA. Set\nB. Tuple"
    assert clean_transcription(" 'NO_QUESTION'. ") == "NO_QUESTION" and clean_transcription("") == ""
    assert clean_transcription("2 nave se apropie frontal.\nA. da.\nB. nu.") == "2 nave se apropie frontal.\nA. da.\nB. nu."


def test_transcribe_image_downscales_the_photo_and_cleans_the_reply(tmp_path):
    settings = make_settings(tmp_path, transcribe_context="Context de test XYZ")
    llm = MockLLM()
    seen = {}
    original = llm.transcribe

    def spy(image, mime, prompt):
        seen["prompt"] = prompt
        return original(image, mime, prompt)
    llm.transcribe = spy
    llm.transcriptions.append("```\nÎntrebare: Care este termenul?\nA. da.\nB. nu.\nC. nu este precizat.\n```")
    big = _jpeg(4000, 3000)
    text, method = transcribe_image(big, llm, settings)
    assert text == "Care este termenul?\nA. da.\nB. nu.\nC. nu este precizat." and method == "llm:mock"
    size, mime = llm.images[0]
    assert mime == "image/jpeg" and size < len(big) and "Context de test XYZ" in seen["prompt"]


# --- OpenAI request shape --------------------------------------------------------

def _provider(handler, model: str = "gpt-6.1-terra", **kw) -> OpenAIProvider:
    provider = OpenAIProvider("sk-test", model, **kw)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test/v1")
    return provider


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 900, "completion_tokens": 40}})


def _rejected(param: str, code: str, message: str) -> httpx.Response:
    return httpx.Response(400, json={"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}})


def test_openai_request_shape_for_gpt6_and_gpt4_models():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _ok('{"status": "answered", "option_labels": ["A"], "answer_text": "x", "explanation": "", "evidence_ids": ["S1"], "confidence": 0.9}')
    assert is_reasoning_model("gpt-6.1-terra") and is_reasoning_model("gpt-5-mini") and is_reasoning_model("o3")
    assert not is_reasoning_model("gpt-4.1-mini") and not is_reasoning_model("gpt-5-chat-latest")
    provider = _provider(handler, vision_model="gpt-6-luna", vision_detail="high", vision_reasoning_effort="none", reasoning_effort="low")
    assert provider.name == "openai:gpt-6.1-terra" and provider.vision_name == "openai:gpt-6-luna"
    provider.transcribe(b"\xff\xd8\xff", "image/jpeg", "PROMPT")
    body = seen[0]
    assert body["model"] == "gpt-6-luna" and body["reasoning_effort"] == "none" and body["temperature"] == 0
    assert body["max_completion_tokens"] == 1200 and "max_tokens" not in body and "service_tier" not in body
    text, image = body["messages"][0]["content"]
    assert text == {"type": "text", "text": "PROMPT"}
    assert image["image_url"]["detail"] == "high" and image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    # the RAG call: low effort, no temperature while reasoning is on, strict JSON schema
    out = provider.generate("sys", "user", {"type": "object"})
    body = seen[1]
    assert out["status"] == "answered" and body["model"] == "gpt-6.1-terra" and body["reasoning_effort"] == "low"
    assert "temperature" not in body and body["response_format"]["json_schema"]["strict"] is True
    # GPT-4.1: no reasoning parameter, deterministic temperature, new token parameter name
    legacy = _provider(handler, model="gpt-4.1-mini", reasoning_effort="low", service_tier="priority")
    legacy.generate("sys", "user", {"type": "object"})
    assert "reasoning_effort" not in seen[2] and seen[2]["temperature"] == 0 and "max_completion_tokens" in seen[2]
    assert seen[2]["service_tier"] == "priority"


def test_openai_drops_rejected_parameters_and_remembers_them():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if body.get("reasoning_effort") == "none":
            return _rejected("reasoning_effort", "unsupported_value", "Unsupported value: 'reasoning_effort' does not support 'none' with this model.")
        if "temperature" in body:
            return _rejected("temperature", "unsupported_parameter", "Unsupported parameter: 'temperature' is not supported with this model.")
        return _ok("Intrebare?\nA. da.\nB. nu.\nC. nu este precizat.")
    provider = _provider(handler, model="gpt-5-mini", vision_reasoning_effort="none")
    assert provider.transcribe(b"\xff\xd8\xff", "image/jpeg", "p").startswith("Intrebare?")
    assert [c.get("reasoning_effort") for c in calls] == ["none", "minimal", "minimal"]
    assert "temperature" in calls[1] and "temperature" not in calls[2]
    provider.transcribe(b"\xff\xd8\xff", "image/jpeg", "p")   # second call: straight to the accepted shape
    assert len(calls) == 4 and calls[3]["reasoning_effort"] == "minimal" and "temperature" not in calls[3]


def test_openai_compatible_servers_get_classic_parameters_and_swap_when_rejected():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "max_tokens" in body:
            return _rejected("max_tokens", "unsupported_parameter", "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
        return _ok('{"status": "answered"}')
    provider = _provider(handler, model="local-model", strict_schema=False, compatible=True)
    assert provider.generate("s", "u", {"type": "object"}) == {"status": "answered"}
    assert "max_tokens" in calls[0] and calls[0]["response_format"] == {"type": "json_object"} and "reasoning_effort" not in calls[0]
    assert "max_completion_tokens" in calls[1] and "max_tokens" not in calls[1]
    provider.generate("s", "u", {"type": "object"})
    assert len(calls) == 3 and "max_completion_tokens" in calls[2]


def test_openai_falls_back_to_json_mode_and_reports_http_errors():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "Invalid schema for response_format 'rokid_answer'.", "type": "invalid_request_error", "param": None, "code": None}})
        return _ok('{"status": "insufficient_context"}')
    provider = _provider(handler)
    assert provider.generate("s", "u", {"type": "object"})["status"] == "insufficient_context"
    assert calls[0]["response_format"]["type"] == "json_schema" and calls[1]["response_format"] == {"type": "json_object"}
    failing = _provider(lambda request: httpx.Response(503, json={"error": {"message": "overloaded"}}))
    with pytest.raises(LLMError, match="HTTP 503 la transcriere"):
        failing.transcribe(b"\xff\xd8\xff", "image/jpeg", "p")
    unsupported = _provider(lambda request: _rejected("foo", "unsupported_parameter", "Unknown parameter: 'foo'."))
    with pytest.raises(LLMError, match="HTTP 400"):
        unsupported.generate("s", "u", {"type": "object"})


# --- vector store and admin status -----------------------------------------------

def test_vectorstore_rebuilds_collection_when_embedding_size_changes(tmp_path):
    store = VectorStore(tmp_path / "chroma")
    store.upsert("questions", "default", ["a"], ["old"], [[0.1] * 512], [{"dataset_id": "d1"}])
    assert store.query("questions", "default", [0.1] * 512, 1)[0].id == "a"
    # provider switched (hash 512 -> openai 1536): the stale collection is dropped and rebuilt
    store.upsert("questions", "default", ["b"], ["new"], [[0.2] * 1536], [{"dataset_id": "d2"}])
    assert [h.id for h in store.query("questions", "default", [0.2] * 1536, 5)] == ["b"] and store.count("questions", "default") == 1
    # a query vector of the old size no longer crashes the pipeline
    assert store.query("questions", "default", [0.1] * 512, 1) == []


def test_admin_status_reports_vision_model_and_stale_embeddings(tmp_path):
    app = create_app(make_settings(tmp_path, vision_model="gpt-6-luna"), llm=MockLLM())
    with TestClient(app) as client:
        status = client.get("/api/v1/admin/status", headers=admin_headers()).json()
        assert status["vision_model"] == "gpt-6-luna" and status["reindex_needed"] == {"documents": [], "qa_datasets": []}
        upload_dataset(client, "python_basics_qa.json", example_dataset_bytes())
        with app.state.db.session() as session:
            for dataset in session.execute(select(QADataset)).scalars():
                dataset.embedding_provider = "openai:text-embedding-3-large"
        status = client.get("/api/v1/admin/status", headers=admin_headers()).json()
        assert status["reindex_needed"]["qa_datasets"] == ["python_basics_qa"]
