"""OpenAI Chat Completions and OpenAI-compatible servers (Gemini's OpenAI
endpoint, Ollama, vLLM, OpenRouter...). Needs an API key with API access; a
ChatGPT subscription does not grant it."""
from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from rokid_backend.llm.base import LLMError, extract_json


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 45.0, max_tokens: int = 700, strict_schema: bool = True) -> None:
        if not api_key:
            raise ValueError("ROKID_LLM_API_KEY is required for the OpenAI provider.")
        if not model:
            raise ValueError("ROKID_LLM_MODEL is required for the OpenAI provider.")
        self.name = f"openai:{model}"
        self.model = model
        self.max_tokens = max_tokens
        self.strict_schema = strict_schema
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.strict_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "rokid_answer", "schema": schema, "strict": True}}
        else:
            body["response_format"] = {"type": "json_object"}
        try:
            response = self._client.post("/chat/completions", json=body)
            if response.status_code == 400 and self.strict_schema:
                body["response_format"] = {"type": "json_object"}
                response = self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Furnizorul LLM a raspuns HTTP {exc.response.status_code}.") from exc
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Apel LLM esuat: {type(exc).__name__}.") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return extract_json(content or "")

    def transcribe(self, image: bytes, mime: str, prompt: str) -> str:
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 1200,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}]}],
        }
        try:
            response = self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Furnizorul LLM a raspuns HTTP {exc.response.status_code} la transcriere.") from exc
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Transcrierea imaginii a esuat: {type(exc).__name__}.") from exc
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return (content or "").strip()
