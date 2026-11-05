"""LLM provider abstraction. Providers return a dict matching ANSWER_SCHEMA;
the pipeline validates it with Pydantic and against the incoming request."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_context"]},
        "option_labels": {"type": "array", "items": {"type": "string"}},
        "answer_text": {"type": "string"},
        "explanation": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["status", "option_labels", "answer_text", "explanation", "evidence_ids", "confidence"],
}


class LLMAnswer(BaseModel):
    status: str = Field(pattern="^(answered|insufficient_context)$")
    option_labels: list[str] = Field(default_factory=list)
    answer_text: str = ""
    explanation: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class VisionProvider(Protocol):
    """Optional capability: providers that can read an image return plain text."""

    def transcribe(self, image: bytes, mime: str, prompt: str) -> str: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction from a model reply (code fences, prose around)."""
    if not text:
        raise LLMError("Raspuns gol de la model.")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            try:
                value = json.loads(candidate[start:end + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise LLMError("Modelul nu a returnat JSON valid.")


def parse_answer(payload: dict[str, Any]) -> LLMAnswer:
    try:
        return LLMAnswer.model_validate({
            "status": str(payload.get("status", "insufficient_context")).strip().lower(),
            "option_labels": [str(x).strip().upper() for x in (payload.get("option_labels") or []) if str(x).strip()],
            "answer_text": str(payload.get("answer_text") or payload.get("text") or "")[:2000],
            "explanation": str(payload.get("explanation") or "")[:2000],
            "evidence_ids": [str(x).strip().upper() for x in (payload.get("evidence_ids") or payload.get("sources") or []) if str(x).strip()],
            "confidence": float(payload.get("confidence") or 0.0) if str(payload.get("confidence", "")).replace(".", "", 1).isdigit() else 0.0,
        })
    except ValidationError as exc:
        raise LLMError(f"Structura raspunsului este invalida: {exc.errors()[0].get('msg')}") from exc
