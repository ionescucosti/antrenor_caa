"""Web-search answer through the built-in `web_search` tool of the OpenAI
Responses API: the model searches, reads the pages and returns the JSON answer
together with the URLs it used. The pipeline runs it in parallel with the
documentation answer for questions that are not in the verified sets."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from rokid_backend.llm.base import ANSWER_SCHEMA, LLMError, extract_json
from rokid_backend.rag.embeddings import quota_exhausted, retry_delay
from rokid_backend.rag.prompts import build_web_input

log = logging.getLogger(__name__)
_RETRY_STATUS = (429, 500, 502, 503, 504)


@dataclass
class WebAnswer:
    status: str = "error"                 # answered | insufficient_context | error
    option_labels: list[str] = field(default_factory=list)
    answer_text: str = ""
    explanation: str = ""
    confidence: float = 0.0
    sources: list[dict[str, str]] = field(default_factory=list)   # {"url", "title"}: cited pages first, then consulted ones
    searches: int = 0
    message: str = ""
    ms: int = 0


class WebSearchProvider(Protocol):
    name: str

    def answer(self, question: str, options: list[tuple[str, str]], instructions: str) -> WebAnswer: ...


def domain_of(url: str) -> str:
    host = (urlsplit(url).netloc or url).lower()
    return host[4:] if host.startswith("www.") else host


def _domains(value: str) -> list[str]:
    out: list[str] = []
    for item in (value or "").split(","):
        item = re.sub(r"^https?://", "", item.strip().lower()).strip("/")
        item = item[4:] if item.startswith("www.") else item   # subdomains are included by the API anyway
        if item and item not in out:
            out.append(item)
    return out[:100]


class OpenAIWebSearch:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 8.0,
                 reasoning_effort: str = "low", context_size: str = "medium", country: str = "RO",
                 allowed_domains: str = "", blocked_domains: str = "", retries: int = 1) -> None:
        if not api_key:
            raise ValueError("Cautarea web are nevoie de cheia OpenAI (OPENAI_API_KEY).")
        if not model:
            raise ValueError("ROKID_WEB_SEARCH_MODEL este obligatoriu.")
        self.name = f"openai-web-search:{model}"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.context_size = context_size if context_size in ("low", "medium", "high") else "medium"
        self.country = (country or "").strip().upper()[:2]
        self.allowed = _domains(allowed_domains)
        self.blocked = _domains(blocked_domains)
        self.retries = retries
        self._structured = True      # text.format json_schema; dropped if the endpoint rejects it
        self._reasoning = True       # reasoning.effort; dropped if the model rejects it
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
                                    headers={"Authorization": f"Bearer {api_key}"})

    # --- request -----------------------------------------------------------

    def _tool(self) -> dict[str, Any]:
        tool: dict[str, Any] = {"type": "web_search", "search_context_size": self.context_size}
        if self.country:
            tool["user_location"] = {"type": "approximate", "country": self.country}
        filters: dict[str, Any] = {}
        if self.allowed:
            filters["allowed_domains"] = self.allowed
        if self.blocked:
            filters["blocked_domains"] = self.blocked
        if filters:
            tool["filters"] = filters
        return tool

    def _body(self, instructions: str, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "instructions": instructions, "input": text, "tools": [self._tool()],
                                "tool_choice": "required", "include": ["web_search_call.action.sources"]}
        if self.reasoning_effort and self._reasoning:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self._structured:
            body["text"] = {"format": {"type": "json_schema", "name": "rokid_answer", "schema": ANSWER_SCHEMA, "strict": True}}
        return body

    def _adapt(self, body: dict[str, Any], response: httpx.Response) -> bool:
        try:
            error = (response.json() or {}).get("error") or {}
        except ValueError:
            error = {}
        message = str(error.get("message") or response.text or "")
        param = str(error.get("param") or "").split(".")[0]
        lowered = message.lower()
        if "text" in body and (param == "text" or "json_schema" in lowered or "text.format" in lowered):
            del body["text"]
            self._structured = False
        elif "reasoning" in body and (param == "reasoning" or "reasoning" in lowered):
            del body["reasoning"]
            self._reasoning = False
        elif body.get("tool_choice") == "required" and "tool_choice" in lowered:
            body["tool_choice"] = "auto"
        else:
            return False
        log.warning("web search endpoint rejected a parameter; retrying without it: %s", message[:160])
        return True

    def answer(self, question: str, options: list[tuple[str, str]], instructions: str) -> WebAnswer:
        started = time.perf_counter()
        body = self._body(instructions, build_web_input(question, options))
        adapted = failures = 0
        while True:
            try:
                response = self._client.post("/responses", json=body)
            except httpx.HTTPError as exc:
                raise LLMError(f"Cautarea web a esuat: {type(exc).__name__}.") from exc
            if response.status_code == 400 and adapted < 3 and self._adapt(body, response):
                adapted += 1
                continue
            if quota_exhausted(response):
                raise LLMError("Contul OpenAI nu mai are credit (insufficient_quota); cautarea web nu poate rula.")
            if response.status_code in _RETRY_STATUS and failures < self.retries:
                failures += 1
                time.sleep(retry_delay(response, failures, cap=2.0))
                continue
            if response.status_code >= 400:
                raise LLMError(f"Cautarea web a raspuns HTTP {response.status_code}.")
            break
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("Cautarea web a intors un raspuns invalid.") from exc
        result = self._parse(data)
        result.ms = int((time.perf_counter() - started) * 1000)
        usage = data.get("usage") or {}
        log.info("web search: model=%s status=%s labels=%s searches=%s sources=%s tokens_in=%s ms=%s", self.model, result.status,
                 ",".join(result.option_labels), result.searches, len(result.sources), usage.get("input_tokens", "?"), result.ms)
        return result

    # --- response ----------------------------------------------------------

    @staticmethod
    def _parse(data: dict[str, Any]) -> WebAnswer:
        text, cited, consulted, searches = "", [], [], 0

        def add(bucket: list[dict[str, str]], url: str, title: str = "") -> None:
            url = (url or "").strip()
            if url and all(s["url"] != url for s in bucket):
                bucket.append({"url": url[:1000], "title": (title or "").strip()[:200]})

        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "web_search_call":
                searches += 1
                for src in (item.get("action") or {}).get("sources") or []:
                    if isinstance(src, dict):
                        add(consulted, str(src.get("url") or ""))
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text += part.get("text") or ""
                        for ann in part.get("annotations") or []:
                            if isinstance(ann, dict) and ann.get("type") == "url_citation":
                                add(cited, str(ann.get("url") or ""), str(ann.get("title") or ""))
        sources = cited + [s for s in consulted if all(c["url"] != s["url"] for c in cited)]
        if not text.strip():
            state = data.get("status") or "fara continut"
            return WebAnswer(status="error", message=f"Cautarea web nu a produs un raspuns ({state}).", searches=searches, sources=sources[:8])
        try:
            payload = extract_json(text)
        except LLMError as exc:
            return WebAnswer(status="error", message=str(exc), searches=searches, sources=sources[:8])
        status = str(payload.get("status") or "insufficient_context").strip().lower()
        labels = [str(x).strip().upper() for x in (payload.get("option_labels") or []) if str(x).strip()]
        answer_text = str(payload.get("answer_text") or "")[:2000]
        try:
            confidence = min(1.0, max(0.0, float(payload.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        if status == "answered" and not (labels or answer_text.strip()):
            status = "insufficient_context"
        return WebAnswer(status="answered" if status == "answered" else "insufficient_context", option_labels=labels,
                         answer_text=answer_text, explanation=str(payload.get("explanation") or "")[:2000], confidence=confidence,
                         sources=sources[:8], searches=searches, message="" if status == "answered" else str(payload.get("explanation") or "")[:300])
