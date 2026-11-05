"""Anthropic Messages API with forced tool use for structured output."""
from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from rokid_backend.llm.base import LLMError


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", base_url: str = "https://api.anthropic.com",
                 timeout: float = 45.0, max_tokens: int = 700, vision_model: str = "") -> None:
        if not api_key:
            raise ValueError("ROKID_LLM_API_KEY is required for the Anthropic provider.")
        self.name = f"anthropic:{model}"
        self.model = model
        self.vision_model = vision_model or model
        self.vision_name = f"anthropic:{self.vision_model}"
        self.max_tokens = max_tokens
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers={
            "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [{"name": "submit_answer", "description": "Submit the structured answer.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": "submit_answer"},
        }
        try:
            response = self._client.post("/v1/messages", json=body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Furnizorul LLM a raspuns HTTP {exc.response.status_code}.") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(f"Apel LLM esuat: {type(exc).__name__}.") from exc
        for block in payload.get("content", []):
            if block.get("type") == "tool_use" and isinstance(block.get("input"), dict):
                return block["input"]
        raise LLMError("Modelul nu a folosit instrumentul de raspuns structurat.")

    def transcribe(self, image: bytes, mime: str, prompt: str) -> str:
        body = {
            "model": self.vision_model, "max_tokens": 1200, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(image).decode()}},
                {"type": "text", "text": prompt}]}],
        }
        try:
            response = self._client.post("/v1/messages", json=body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Furnizorul LLM a raspuns HTTP {exc.response.status_code} la transcriere.") from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise LLMError(f"Transcrierea imaginii a esuat: {type(exc).__name__}.") from exc
        return "".join(block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text").strip()
