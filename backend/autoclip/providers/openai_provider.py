"""OpenAI-compatible provider.

Deliberately the widest adapter in AutoClip. Because ``base_url`` is
configurable and the Chat Completions shape is a de-facto standard, this one
class also covers OpenRouter, Groq, DeepSeek, Together, Fireworks, vLLM, and a
local LM Studio server. "Bring any API key" is mostly this file.
"""

from __future__ import annotations

import logging

from .base import DetectionConfig, LLMProvider, ProviderError, ProviderStatus

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"

#: Explicit output-token budget for a window's JSON response. Without it,
#: endpoints apply a low default cap (often 4096), and a response truncated
#: mid-JSON fails the endpoint's own JSON-mode validation — the
#: "max completion tokens reached before generating a valid document" error.
MAX_OUTPUT_TOKENS = 8192

SUGGESTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
]

#: Ready-made endpoints for the settings UI. Users can enter any other URL.
KNOWN_ENDPOINTS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Groq": "https://api.groq.com/openai/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Together": "https://api.together.xyz/v1",
    "LM Studio (local)": "http://localhost:1234/v1",
}


class OpenAIProvider(LLMProvider):
    name = "openai"
    requires_key = True

    def __init__(self, model: str = "", *, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model or DEFAULT_MODEL, api_key=api_key, base_url=base_url)

    def _client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - openai is a core dep
            raise ProviderError("The openai package is not installed.", provider=self.name) from exc

        # Local servers ignore the key but the SDK still requires a non-empty
        # value, so supply a placeholder rather than failing the request.
        key = self.api_key or ("not-needed" if self._is_local() else None)
        if not key:
            raise ProviderError(
                "No API key is set for the OpenAI-compatible provider.",
                provider=self.name,
                hint="Add one with `autoclip config set-secret openai`.",
            )

        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    def _is_local(self) -> bool:
        return bool(self.base_url) and (
            "localhost" in self.base_url or "127.0.0.1" in self.base_url
        )

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        client = self._client()
        request = {
            "model": self.model,
            "temperature": config.temperature,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # response_format is only reliably honoured by OpenAI's own endpoint:
        # other vendors 400 on it, and some models fail their own JSON-mode
        # validation outright ("json_validate_failed"). A plain request works
        # everywhere — the shared detection loop already extracts and repairs
        # JSON from freeform responses.
        try:
            return await self._send(client, request, json_mode=True)
        except ProviderError as exc:
            if _is_rejected_parameter(exc):
                log.debug("Endpoint rejected a request parameter; retrying plainly.")
                plain = {k: v for k, v in request.items() if k != "max_tokens"}
                return await self._send(client, plain, json_mode=False)
            # An output-cap failure is request-side: no retry shape fixes it.
            if _is_output_token_cap(exc):
                raise
            if _is_json_mode_failure(exc):
                log.debug("JSON mode failed validation; retrying without response_format.")
                try:
                    return await self._send(client, request, json_mode=False)
                except ProviderError:
                    raise exc from None  # report the original, more specific failure
            raise

    async def _send(self, client, request: dict, *, json_mode: bool) -> str:
        """Make one chat completion call, translating errors to ProviderError."""
        kwargs = dict(request)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _translate(exc, self.name, self.model) from exc

        choices = response.choices or []
        if not choices:
            raise ProviderError("The endpoint returned no choices.", provider=self.name)
        return choices[0].message.content or ""

    async def health_check(self) -> ProviderStatus:
        if not self.api_key and not self._is_local():
            return ProviderStatus(
                name=self.name, available=False, detail="No API key set", models=SUGGESTED_MODELS
            )
        try:
            client = self._client()
            models = await client.models.list()
            names = sorted(m.id for m in models.data)[:50]
        except Exception as exc:
            return ProviderStatus(name=self.name, available=False, detail=str(exc)[:200])
        return ProviderStatus(
            name=self.name,
            available=True,
            detail=self.base_url or "https://api.openai.com/v1",
            models=names or SUGGESTED_MODELS,
        )


def _is_rejected_parameter(exc: Exception) -> bool:
    """The endpoint refused part of the request itself (parameter unsupported)."""
    lowered = str(exc).lower()
    return (
        "max_tokens" in lowered
        or "response_format" in lowered
        or "unsupported" in lowered
        or "unrecognized" in lowered
    )


def _is_json_mode_failure(exc: Exception) -> bool:
    """The request was accepted, but the endpoint's JSON mode failed."""
    lowered = str(exc).lower()
    return (
        "json_validate_failed" in lowered
        or "failed to generate json" in lowered
        or "failed to validate json" in lowered
    )


def _is_output_token_cap(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "max completion tokens" in lowered or "max output tokens" in lowered


def _translate(exc: Exception, provider: str, model: str) -> ProviderError:
    message = str(exc)
    lowered = message.lower()

    if "401" in lowered or "invalid api key" in lowered or "incorrect api key" in lowered:
        return ProviderError(
            "The endpoint rejected the API key.",
            provider=provider,
            hint="Re-add it with `autoclip config set-secret openai`.",
        )
    if "429" in lowered or "rate limit" in lowered:
        return ProviderError(
            "Rate limit reached.",
            provider=provider,
            hint="Wait and retry, or switch providers.",
        )
    if "404" in lowered or "does not exist" in lowered or "model_not_found" in lowered:
        return ProviderError(
            f"The endpoint does not recognise the model '{model}'.",
            provider=provider,
            hint="Check the model name against your provider's catalogue.",
        )
    if "connection" in lowered or "connect" in lowered:
        return ProviderError(
            "Could not reach the endpoint.",
            provider=provider,
            hint="Check the base URL in settings, and that any local server is running.",
        )
    if "quota" in lowered or "billing" in lowered:
        return ProviderError("The account has no remaining quota.", provider=provider)
    if "max completion tokens" in lowered or "max output tokens" in lowered:
        return ProviderError(
            "The model ran out of output tokens before finishing its JSON response.",
            provider=provider,
            hint=(
                "Try a model with a larger output budget, or lower the maximum "
                "clip count in settings so each window returns less JSON."
            ),
        )

    return ProviderError(f"Request failed: {message}", provider=provider)
