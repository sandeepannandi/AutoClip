"""LLM providers for highlight detection.

Four adapters behind one interface. ``openai`` is the widest of them: because
its base URL is configurable, it also serves OpenRouter, Groq, DeepSeek,
Together, and any local OpenAI-compatible server.
"""

from __future__ import annotations

import logging

from ..config import Settings, get_secret
from .anthropic_provider import AnthropicProvider
from .base import (
    ClipCandidate,
    ClipCandidates,
    DetectionConfig,
    LLMProvider,
    ProviderError,
    ProviderStatus,
    TranscriptWindow,
)
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

log = logging.getLogger(__name__)

__all__ = [
    "PROVIDERS",
    "AnthropicProvider",
    "ClipCandidate",
    "ClipCandidates",
    "DetectionConfig",
    "GeminiProvider",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderStatus",
    "TranscriptWindow",
    "build_provider",
    "detection_config",
    "provider_names",
]

PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


def provider_names() -> list[str]:
    return list(PROVIDERS)


def build_provider(name: str | None = None, settings: Settings | None = None) -> LLMProvider:
    """Construct a configured provider.

    Pulls the model and base URL from settings and the API key from the keyring,
    so callers never handle secrets themselves.
    """
    from ..config import load

    settings = settings if settings is not None else load()
    key = name or settings.active_provider

    provider_cls = PROVIDERS.get(key)
    if provider_cls is None:
        raise ProviderError(
            f"Unknown provider '{key}'.",
            hint=f"Available providers: {', '.join(PROVIDERS)}",
        )

    provider_settings = settings.provider(key)
    api_key = get_secret(key, settings) if provider_cls.requires_key else None

    return provider_cls(
        provider_settings.model,
        api_key=api_key,
        base_url=provider_settings.base_url,
    )


def detection_config(settings: Settings | None = None) -> DetectionConfig:
    """Build a :class:`DetectionConfig` from user settings.

    When the account has a logged posting history, the config carries a few-shot
    block of the account's own best and worst performers so detection can
    calibrate to this creator's audience. A failure to build one must never
    fail detection — the block is simply omitted.
    """
    from ..config import load
    from ..pipeline.outcomes import few_shot_examples_from_db, render_few_shot_block

    settings = settings if settings is not None else load()
    few_shot_block = ""
    if settings.tracking.learn_from_outcomes:
        try:
            few_shot_block = render_few_shot_block(few_shot_examples_from_db())
        except Exception:
            log.exception("Few-shot example build failed; continuing without them.")

    return DetectionConfig(
        min_duration_s=settings.clips.min_duration_s,
        max_duration_s=settings.clips.max_duration_s,
        max_clips=settings.clips.max_clips,
        language=settings.whisper.language,
        few_shot_block=few_shot_block,
    )
