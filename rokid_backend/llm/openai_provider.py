"""OpenAI Chat Completions and OpenAI-compatible servers (Gemini's OpenAI
endpoint, Ollama, vLLM, OpenRouter...). Needs an API key with API access; a
ChatGPT subscription does not grant it.

Request parameters differ between model generations: GPT-4.1 accepts
`temperature`, GPT-5/GPT-6 reasoning models reject it unless reasoning is off,
`max_tokens` was replaced by `max_completion_tokens`, and `reasoning_effort:
none` exists only on newer models. The provider sends the best guess for the
model family; when the endpoint answers HTTP 400 for a parameter it removes or
downgrades that parameter, retries, and remembers the outcome for the rest of
the process, so only the first call pays for the extra round trip."""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any

import httpx

from rokid_backend.llm.base import LLMError, extract_json
from rokid_backend.rag.embeddings import quota_exhausted, retry_delay

log = logging.getLogger(__name__)

VISION_MAX_TOKENS = 1200   # a question with its options is 100-300 tokens; the rest is headroom for reasoning tokens
_REASONING_FAMILIES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
_EFFORT_DOWNGRADE = {"none": "minimal", "minimal": "low"}
_PARAMS = ("max_completion_tokens", "max_tokens", "reasoning_effort", "response_format", "service_tier", "temperature", "detail")
_RETRY_STATUS = (429, 500, 502, 503, 504)   # transient: retried with a short backoff
_BACKOFF_SECONDS = 0.4


def is_reasoning_model(model: str) -> bool:
    name = (model or "").lower()
    return name.startswith(_REASONING_FAMILIES) and "chat-latest" not in name


def _set_detail(body: dict[str, Any], detail: str) -> None:
    for message in body.get("messages", []):
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    part["image_url"]["detail"] = detail


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 45.0, max_tokens: int = 700, strict_schema: bool = True, compatible: bool = False,
                 reasoning_effort: str = "", vision_model: str = "", vision_detail: str = "high",
                 vision_reasoning_effort: str = "", service_tier: str = "", retries: int = 2) -> None:
        if not api_key:
            raise ValueError("ROKID_LLM_API_KEY is required for the OpenAI provider.")
        if not model:
            raise ValueError("ROKID_LLM_MODEL is required for the OpenAI provider.")
        self.name = f"openai:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.strict_schema = strict_schema
        self.reasoning_effort = reasoning_effort
        self.vision_model = vision_model or model
        self.vision_name = f"openai:{self.vision_model}"
        self.vision_detail = vision_detail or "high"
        self.vision_reasoning_effort = vision_reasoning_effort
        self.service_tier = service_tier
        self.retries = retries                  # extra attempts on timeouts, connection errors and HTTP 429/5xx
        self._legacy_max_tokens = compatible    # third-party servers: the classic parameter name
        self._dropped: set[str] = set()         # "<model>:temperature", "service_tier": rejected by the endpoint
        self._effort: dict[str, str] = {}       # per model: effort accepted after a downgrade ("" = not sent)
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})

    # --- request shape -------------------------------------------------------

    def _effort_for(self, model: str, configured: str) -> str:
        effort = self._effort.get(model, configured)
        return effort if effort and is_reasoning_model(model) else ""

    def _body(self, model: str, messages: list[dict[str, Any]], max_tokens: int, effort: str) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        if f"{model}:temperature" not in self._dropped and (not is_reasoning_model(model) or effort == "none"):
            body["temperature"] = 0
        body["max_tokens" if self._legacy_max_tokens else "max_completion_tokens"] = max_tokens
        if effort:
            body["reasoning_effort"] = effort
        if self.service_tier and "service_tier" not in self._dropped:
            body["service_tier"] = self.service_tier
        return body

    def _adapt(self, body: dict[str, Any], response: httpx.Response) -> bool:
        """Remove or downgrade the parameter a 400 complains about. False when
        the error is not about a request parameter we control."""
        try:
            error = (response.json() or {}).get("error") or {}
        except ValueError:
            error = {}
        message = str(error.get("message") or response.text or "")
        param = str(error.get("param") or "").split(".")[0]
        code = str(error.get("code") or "")
        if param not in _PARAMS:
            lowered = message.lower()
            param = next((p for p in _PARAMS if re.search(rf"\b{p}\b", lowered)), "")
        model = str(body.get("model") or "")
        if param == "temperature" and "temperature" in body:
            del body["temperature"]
            self._dropped.add(f"{model}:temperature")
        elif param == "max_tokens" and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
            self._legacy_max_tokens = False
        elif param == "max_completion_tokens" and "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
            self._legacy_max_tokens = True
        elif param == "reasoning_effort" and "reasoning_effort" in body:
            lower = "" if code == "unsupported_parameter" else _EFFORT_DOWNGRADE.get(body["reasoning_effort"], "")
            if lower:
                body["reasoning_effort"] = lower
            else:
                del body["reasoning_effort"]
            self._effort[model] = lower
        elif param == "service_tier" and "service_tier" in body:
            del body["service_tier"]
            self._dropped.add("service_tier")
        elif param == "detail" and self.vision_detail != "high":
            self.vision_detail = "high"
            _set_detail(body, "high")
        elif (body.get("response_format") or {}).get("type") == "json_schema":
            body["response_format"] = {"type": "json_object"}   # schema too strict for this server: plain JSON mode
            self.strict_schema = False
            param = "response_format"
        else:
            return False
        log.warning("LLM endpoint rejected '%s' for %s; retrying without it: %s", param, model, message[:160])
        return True

    def _post(self, body: dict[str, Any], what: str) -> dict[str, Any]:
        suffix = f" la {what}" if what else ""
        adapted = failures = 0
        while True:
            try:
                response = self._client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                failures += 1
                if failures > self.retries:
                    raise LLMError(f"Apel LLM esuat{suffix}: {type(exc).__name__}.") from exc
                log.warning("LLM call failed with %s; retry %d/%d", type(exc).__name__, failures, self.retries)
                time.sleep(retry_delay(None, failures, _BACKOFF_SECONDS))
                continue
            if response.status_code == 400 and adapted < 5 and self._adapt(body, response):
                adapted += 1
                continue
            if quota_exhausted(response):
                raise LLMError(f"Contul OpenAI nu mai are credit (insufficient_quota){suffix}; adauga credit in platform.openai.com > Billing.")
            if response.status_code in _RETRY_STATUS and failures < self.retries:
                failures += 1
                log.warning("LLM endpoint answered HTTP %s; retry %d/%d", response.status_code, failures, self.retries)
                time.sleep(retry_delay(response, failures, _BACKOFF_SECONDS))
                continue
            if response.status_code >= 400:
                raise LLMError(f"Furnizorul LLM a raspuns HTTP {response.status_code}{suffix}.")
            try:
                return response.json()
            except ValueError as exc:
                raise LLMError(f"Apel LLM esuat{suffix}: raspuns invalid.") from exc

    @staticmethod
    def _content(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Apel LLM esuat: raspuns fara continut.") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""

    # --- public ----------------------------------------------------------------

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        body = self._body(self.model, messages, self.max_tokens, self._effort_for(self.model, self.reasoning_effort))
        if self.strict_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "rokid_answer", "schema": schema, "strict": True}}
        else:
            body["response_format"] = {"type": "json_object"}
        return extract_json(self._content(self._post(body, "")))

    def transcribe(self, image: bytes, mime: str, prompt: str) -> str:
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        model = self.vision_model
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url, "detail": self.vision_detail}}]}]
        body = self._body(model, messages, VISION_MAX_TOKENS, self._effort_for(model, self.vision_reasoning_effort))
        started = time.perf_counter()
        payload = self._post(body, "transcriere")
        content = self._content(payload).strip()
        choice = (payload.get("choices") or [{}])[0] or {}
        usage = payload.get("usage") or {}
        log.info("vision transcription: model=%s effort=%s detail=%s tokens_in=%s tokens_out=%s ms=%.0f finish=%s",
                 model, body.get("reasoning_effort", "-"), self.vision_detail, usage.get("prompt_tokens", "?"),
                 usage.get("completion_tokens", "?"), (time.perf_counter() - started) * 1000, choice.get("finish_reason", "?"))
        if choice.get("finish_reason") == "length":
            log.warning("vision transcription hit the %s-token limit; the text may be incomplete", VISION_MAX_TOKENS)
        return content
