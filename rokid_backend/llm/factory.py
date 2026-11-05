from __future__ import annotations

from rokid_backend.llm.anthropic_provider import AnthropicProvider
from rokid_backend.llm.base import LLMProvider
from rokid_backend.llm.mock import MockLLM
from rokid_backend.llm.openai_provider import OpenAIProvider


def build_llm_provider(settings) -> LLMProvider:
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLM()
    key, model = settings.llm_key, settings.llm_model_name
    if provider == "openai":
        return OpenAIProvider(key, model, settings.llm_base_url or "https://api.openai.com/v1",
                              settings.llm_timeout_seconds, settings.llm_max_tokens, strict_schema=True)
    if provider == "openai_compatible":
        if not settings.llm_base_url:
            raise ValueError("ROKID_LLM_BASE_URL is required for openai_compatible.")
        return OpenAIProvider(key or "none", model, settings.llm_base_url, settings.llm_timeout_seconds,
                              settings.llm_max_tokens, strict_schema=False)
    if provider == "anthropic":
        return AnthropicProvider(key, model, settings.llm_base_url or "https://api.anthropic.com",
                                 settings.llm_timeout_seconds, settings.llm_max_tokens)
    raise ValueError(f"Unknown LLM provider {provider}")
