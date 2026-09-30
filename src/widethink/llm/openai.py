"""GPT models and open models behind OpenAI-compatible servers (vLLM, Ollama, ...).

Install with ``pip install 'widethink[openai]'``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from widethink.errors import ConfigurationError, RefusalError, TruncatedOutputError
from widethink.llm.base import LLMRequest, LLMResponse, Usage
from widethink.llm.jsonschema import extract_json

StructuredMode = Literal["json_schema", "json_object", "prompt"]

_SCHEMA_INSTRUCTIONS = (
    "\n\nReply with a single JSON object and nothing else. "
    "It must follow this JSON schema:\n{schema}"
)
_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


class OpenAICompatibleLLM:
    """Chat Completions API of OpenAI or of any compatible server.

    Args:
        model: Model name as the server knows it.
        client: A ready ``openai.AsyncOpenAI`` (or compatible) client.
        base_url: Server URL for local or third-party endpoints.
        api_key: Key for the endpoint; the SDK falls back to ``OPENAI_API_KEY``.
        structured: ``json_schema`` uses strict structured outputs;
            ``json_object`` uses JSON mode and puts the schema in the prompt;
            ``prompt`` relies on the prompt alone (servers without JSON support).
        max_tokens_param: Name of the output limit parameter; older servers
            accept only ``max_tokens``.
        send_effort: Forward the effort hint as ``reasoning_effort`` (for
            reasoning models that support it).
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        structured: StructuredMode = "json_schema",
        max_tokens_param: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
        send_effort: bool = False,
    ) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ConfigurationError(
                    "OpenAICompatibleLLM needs the 'openai' package: "
                    "pip install 'widethink[openai]'"
                ) from error
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._client = client
        self.model = model
        self.base_url = base_url
        self.structured = structured
        self.max_tokens_param = max_tokens_param
        self.send_effort = send_effort

    @property
    def name(self) -> str:
        return (
            f"openai:{self.model}" if self.base_url is None else f"openai-compatible:{self.model}"
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        completion = await self._client.chat.completions.create(**self.build_params(request))
        choice = completion.choices[0]
        usage = _usage(getattr(completion, "usage", None))
        model = str(getattr(completion, "model", None) or self.model)
        if choice.finish_reason == "length":
            raise TruncatedOutputError(
                f"{model} reached the output limit of {request.max_tokens} tokens",
                usage=usage,
                model=model,
            )
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise RefusalError(f"{model} declined the request: {refusal}", usage=usage, model=model)
        text = choice.message.content or ""
        data: Any = None
        if request.json_schema is not None:
            try:
                data = extract_json(text)
            except ValueError:
                data = None
        return LLMResponse(
            text=text, data=data, usage=usage, model=model, stop_reason=choice.finish_reason
        )

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        """Translate a provider-neutral request into ``chat.completions.create`` arguments."""
        system = request.system
        if request.json_schema is not None and self.structured != "json_schema":
            schema = json.dumps(request.json_schema, ensure_ascii=False)
            system = system + _SCHEMA_INSTRUCTIONS.format(schema=schema)
        messages: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self.max_tokens_param: request.max_tokens,
        }
        if request.json_schema is not None:
            if self.structured == "json_schema":
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _NAME_UNSAFE.sub("_", request.schema_name)[:64] or "response",
                        "schema": request.json_schema,
                        "strict": True,
                    },
                }
            elif self.structured == "json_object":
                params["response_format"] = {"type": "json_object"}
        if request.effort is not None and self.send_effort:
            params["reasoning_effort"] = request.effort
        if request.temperature is not None:
            params["temperature"] = request.temperature
        return params


def _usage(raw: Any) -> Usage:
    """Normalize OpenAI usage: ``prompt_tokens`` already includes cached tokens."""
    if raw is None:
        return Usage()
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    completion_details = getattr(raw, "completion_tokens_details", None)
    return Usage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        cache_read_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0),
        reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    )
