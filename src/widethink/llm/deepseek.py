"""DeepSeek through its OpenAI-compatible API (``pip install 'widethink[openai]'``).

DeepSeek differs from OpenAI in ways that matter to the harness (API docs,
September 2026):

* **Models** are ``deepseek-flash`` (fast and cheap) and ``deepseek-v4-pro``.
* **Thinking is on by default.** The harness makes many short structured steps,
  so this provider turns thinking off unless asked (``thinking=True``, used for
  the reasoning-mode baseline, where the effort hint becomes DeepSeek's
  ``reasoning_effort``).
* **Structured output is JSON mode only** (``json_object``, no strict schemas),
  and the prompt must contain the word "json" and an example; the schema and a
  generated example are appended to the system prompt.
* The output limit is ``max_tokens``; cache hits are reported in the usage and
  are much cheaper, which the constant system prompt of each step exploits.
* ``insufficient_system_resource`` / ``aborted`` finishes are retried.

There is no embeddings endpoint: pair this provider with the default hashing
embedder or a local one (``widethink[local]``).
"""

from __future__ import annotations

import os
from typing import Any

from widethink.errors import ConfigurationError
from widethink.llm.base import LLMRequest
from widethink.llm.openai import OpenAICompatibleLLM

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
KEY_VARIABLE = "DEEPSEEK_API_KEY"

#: The harness's effort hint mapped onto DeepSeek's reasoning efforts.
_EFFORT = {"low": "low", "medium": "high", "high": "high"}


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek chat models with the settings the harness needs.

    Args:
        model: ``deepseek-flash`` (default) or ``deepseek-v4-pro``.
        thinking: Enable DeepSeek's thinking mode for every call.
        api_key: Defaults to the ``DEEPSEEK_API_KEY`` environment variable.
        base_url: DeepSeek's endpoint; override for proxies.
        client: A ready OpenAI-compatible client (tests).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        thinking: bool = False,
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        client: Any = None,
    ) -> None:
        key = api_key or os.environ.get(KEY_VARIABLE)
        if client is None and not key:
            raise ConfigurationError(
                f"DeepSeekLLM needs an API key: set {KEY_VARIABLE} or pass api_key="
            )
        super().__init__(
            model,
            client=client,
            base_url=base_url,
            api_key=key,
            structured="json_object",
            max_tokens_param="max_tokens",
        )
        self.thinking = thinking

    @property
    def name(self) -> str:
        return f"deepseek:{self.model}" + ("+thinking" if self.thinking else "")

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        params = super().build_params(request)
        thinking: dict[str, str] = {"type": "enabled" if self.thinking else "disabled"}
        if self.thinking and request.effort is not None:
            thinking["reasoning_effort"] = _EFFORT[request.effort]
        params["extra_body"] = {**params.get("extra_body", {}), "thinking": thinking}
        return params
