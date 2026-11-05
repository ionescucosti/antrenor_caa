"""Deterministic provider for tests and development. It never calls the network.

Behaviour: if scripted replies were queued they are returned in order;
otherwise a simple lexical heuristic picks the option best supported by the
evidence, or abstains."""
from __future__ import annotations

import re
from typing import Any

from rokid_backend.core.text import normalize_text


class MockLLM:
    name = "mock"

    def __init__(self) -> None:
        self.scripted: list[dict[str, Any]] = []
        self.calls: list[dict[str, str]] = []
        self.transcriptions: list[str] = []
        self.images: list[tuple[int, str]] = []

    def script(self, *responses: dict[str, Any]) -> None:
        self.scripted.extend(responses)

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if self.scripted:
            return self.scripted.pop(0)
        return self._heuristic(user)

    def transcribe(self, image: bytes, mime: str, prompt: str) -> str:
        from rokid_backend.llm.base import LLMError
        self.images.append((len(image), mime))
        if self.transcriptions:
            return self.transcriptions.pop(0)
        raise LLMError("Furnizorul mock nu poate citi imagini; configureaza openai sau anthropic.")

    @staticmethod
    def _heuristic(user: str) -> dict[str, Any]:
        evidence = re.findall(r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\[S\d+\]|\n=== END EVIDENCE ===)", user, re.DOTALL)
        options = re.findall(r"^\s*([A-L])\. (.+)$", user.split("OPTIONS:", 1)[1].split("\n\n", 1)[0] if "OPTIONS:" in user else "", re.MULTILINE)
        if not evidence:
            return {"status": "insufficient_context", "option_labels": [], "answer_text": "", "explanation": "no evidence", "evidence_ids": [], "confidence": 0.0}
        corpus = {eid: normalize_text(text) for eid, text in evidence}
        if options:
            best, best_score, best_eid = None, 0, None
            for label, text in options:
                words = [w for w in normalize_text(text).split() if len(w) > 2]
                for eid, body in corpus.items():
                    score = sum(body.count(w) for w in words) + (5 if normalize_text(text) in body else 0)
                    if score > best_score:
                        best, best_score, best_eid = (label, text), score, eid
            if best is None or best_score == 0:
                return {"status": "insufficient_context", "option_labels": [], "answer_text": "", "explanation": "evidence does not mention any option", "evidence_ids": [], "confidence": 0.1}
            return {"status": "answered", "option_labels": [best[0]], "answer_text": best[1],
                    "explanation": f"Supported by {best_eid}.", "evidence_ids": [best_eid], "confidence": 0.7}
        eid, text = evidence[0]
        return {"status": "answered", "option_labels": [], "answer_text": text.strip()[:300],
                "explanation": f"Taken from {eid}.", "evidence_ids": [eid], "confidence": 0.5}
