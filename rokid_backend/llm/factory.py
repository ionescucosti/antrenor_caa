from __future__ import annotations

from rokid_backend.llm.anthropic_provider import AnthropicProvider
from rokid_backend.llm.base import LLMProvider
from rokid_backend.llm.mock import MockLLM
from rokid_backend.llm.openai_provider import OpenAIProvider
from rokid_backend.llm.web_search import OpenAIWebSearch, WebSearchProvider


def _vision_kwargs(settings) -> dict:
    return dict(reasoning_effort=settings.llm_reasoning_effort, vision_model=settings.vision_model,
                vision_detail=settings.vision_detail, vision_reasoning_effort=settings.vision_reasoning_effort,
                service_tier=settings.llm_service_tier)


def build_llm_provider(settings) -> LLMProvider:
    provider = settings.llm_provider
    if provider == "mock":
        return MockLLM()
    key, model = settings.llm_key, settings.llm_model_name
    if provider == "openai":
        return OpenAIProvider(key, model, settings.llm_base_url or "https://api.openai.com/v1",
                              settings.llm_timeout_seconds, settings.llm_max_tokens, strict_schema=True, **_vision_kwargs(settings))
    if provider == "openai_compatible":
        if not settings.llm_base_url:
            raise ValueError("ROKID_LLM_BASE_URL is required for openai_compatible.")
        return OpenAIProvider(key or "none", model, settings.llm_base_url, settings.llm_timeout_seconds,
                              settings.llm_max_tokens, strict_schema=False, compatible=True, **_vision_kwargs(settings))
    if provider == "anthropic":
        return AnthropicProvider(key, model, settings.llm_base_url or "https://api.anthropic.com",
                                 settings.llm_timeout_seconds, settings.llm_max_tokens, vision_model=settings.vision_model)
    raise ValueError(f"Unknown LLM provider {provider}")


def build_web_search_provider(settings) -> WebSearchProvider | None:
    """None when web search is disabled or the LLM provider is not OpenAI: the
    web_search tool lives in the OpenAI Responses API and needs its key."""
    if not settings.web_search_enabled or settings.llm_provider != "openai" or not settings.web_search_model:
        return None
    key = settings.llm_key
    if not key:
        return None
    return OpenAIWebSearch(key, settings.web_search_model, settings.llm_base_url or "https://api.openai.com/v1",
                           settings.web_search_timeout_seconds, settings.web_search_reasoning_effort, settings.web_search_context_size,
                           settings.web_search_country, settings.web_search_allowed_domains, settings.web_search_blocked_domains)
