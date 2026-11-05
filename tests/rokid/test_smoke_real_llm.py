"""Optional live-provider smoke test. Runs only when ROKID_SMOKE_LLM=1 and the
provider credentials are configured; it is never part of the deterministic suite."""
import os

import pytest

from rokid_backend.core.config import Settings
from rokid_backend.llm.base import ANSWER_SCHEMA, parse_answer
from rokid_backend.llm.factory import build_llm_provider
from rokid_backend.rag.prompts import SYSTEM_PROMPT, Evidence, build_user_prompt

pytestmark = pytest.mark.skipif(os.environ.get("ROKID_SMOKE_LLM") != "1", reason="set ROKID_SMOKE_LLM=1 to run against a real provider")


def test_real_provider_structured_answer():
    settings = Settings()
    assert settings.llm_provider != "mock", "configure ROKID_LLM_PROVIDER / ROKID_LLM_API_KEY"
    provider = build_llm_provider(settings)
    evidence = [Evidence(id="S1", kind="document", title="guide.md", text="A tuple is an immutable sequence. Lists and dicts are mutable.", similarity=0.9)]
    prompt = build_user_prompt("Which Python type is immutable?", "multiple_choice", [("A", "List"), ("B", "Tuple"), ("C", "Dict")], evidence)
    parsed = parse_answer(provider.generate(SYSTEM_PROMPT, prompt, ANSWER_SCHEMA))
    assert parsed.status == "answered" and parsed.option_labels == ["B"] and "S1" in parsed.evidence_ids
