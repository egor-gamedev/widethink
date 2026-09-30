"""Claude through the official Anthropic SDK (``pip install 'widethink[anthropic]'``)."""

from __future__ import annotations

import json
from typing import Any

from widethink.errors import ConfigurationError, RefusalError, TruncatedOutputError
from widethink.llm.base import LLMRequest, LLMResponse, Usage

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Claude models via ``anthropic.AsyncAnthropic``.

    * Structured output uses ``output_config.format`` with a JSON schema, so a
      completed reply is always valid JSON.
    * Sampling parameters are sent only when a request sets them: current Claude
      models reject ``temperature``. Diversity in widethink comes from the
      harness's lottery, not from sampling noise.
    * ``fallbacks=True`` (the default) opts into server-side fallback
      (``fallbacks="default"``): a request declined by a model's safety
      classifiers is re-run on Anthropic's recommended substitute. The model
      that actually answered is reported in :attr:`LLMResponse.model` and ends
      up in run records, so benchmark results stay attributable.
    * The system prompt carries a cache breakpoint: it holds the stable part of
      every step (rules, task, context index), so later steps read it from cache.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any = None,
        fallbacks: bool = True,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ConfigurationError(
                    "AnthropicLLM needs the 'anthropic' package: pip install 'widethink[anthropic]'"
                ) from error
            options: dict[str, Any] = {}
            if max_retries is not None:
                options["max_retries"] = max_retries
            if timeout is not None:
                options["timeout"] = timeout
            client = anthropic.AsyncAnthropic(**options)
        self._client = client
        self.model = model
        self.fallbacks = fallbacks

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        params = self.build_params(request)
        if self.fallbacks:
            message = await self._client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **params
            )
        else:
            message = await self._client.messages.create(**params)

        usage = _usage(message.usage)
        model = str(getattr(message, "model", None) or self.model)
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            raise RefusalError(
                f"{model} declined the request (category: {category})", usage=usage, model=model
            )
        if stop_reason == "max_tokens":
            raise TruncatedOutputError(
                f"{model} reached max_tokens={request.max_tokens}", usage=usage, model=model
            )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        return LLMResponse(
            text=text,
            data=_json_or_none(text) if request.json_schema is not None else None,
            usage=usage,
            model=model,
            stop_reason=stop_reason,
        )

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Translate a provider-neutral request into ``messages.create`` arguments."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system:
            block: dict[str, Any] = {"type": "text", "text": request.system}
            if request.cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            params["system"] = [block]
        output_config: dict[str, Any] = {}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if request.effort is not None:
            output_config["effort"] = request.effort
        if output_config:
            params["output_config"] = output_config
        if request.temperature is not None:
            params["temperature"] = request.temperature
        return params


def _usage(raw: Any) -> Usage:
    """Normalize Anthropic usage: cached prompt tokens are reported apart from ``input_tokens``."""
    uncached = int(getattr(raw, "input_tokens", 0) or 0)
    read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
    written = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    return Usage(
        input_tokens=uncached + read + written,
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        cache_read_tokens=read,
        cache_write_tokens=written,
    )


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
