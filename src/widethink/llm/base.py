"""Provider-neutral interface to language models.

A provider is anything with a ``name`` and an async ``generate`` method (see
:class:`LLM`). The harness never talks to an SDK directly, which keeps the
thinking process identical across Claude, GPT and open models, and lets tests
run against deterministic fakes.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

Role = Literal["user", "assistant"]
Effort = Literal["low", "medium", "high"]


class Message(BaseModel):
    """One turn of a conversation."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class Usage(BaseModel):
    """Token usage of one or more calls, normalized across providers.

    ``input_tokens`` counts *every* prompt token the model processed, including
    tokens read from or written to a prompt cache: Anthropic reports those
    separately, OpenAI includes them, and the providers here normalize both to
    this definition. ``cache_read_tokens`` and ``cache_write_tokens`` are subsets
    of ``input_tokens``; ``reasoning_tokens`` is the part of ``output_tokens``
    spent on hidden reasoning when the provider reports it.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class LLMRequest(BaseModel):
    """A single model call, described independently of any provider."""

    model_config = ConfigDict(frozen=True)

    system: str
    messages: tuple[Message, ...]
    max_tokens: int
    json_schema: dict[str, Any] | None = None
    """Strict JSON schema the reply must follow; providers use their native
    structured-output feature when they have one."""
    schema_name: str = "response"
    effort: Effort | None = None
    """Reasoning effort hint; providers that support it map it to their own knob."""
    temperature: float | None = None
    """Sent only when set: several current models reject sampling parameters."""
    purpose: str = "generic"
    """What the call is for (``expand``, ``critic``, ``synthesis`` ...) - used in accounting."""
    cache_system: bool = True
    """Mark the system prompt as a cacheable prefix where the provider supports it."""

    @classmethod
    def single(cls, system: str, user: str, *, max_tokens: int, **fields: Any) -> LLMRequest:
        """Build a request with one user message."""
        return cls(
            system=system,
            messages=(Message(role="user", content=user),),
            max_tokens=max_tokens,
            **fields,
        )


class LLMResponse(BaseModel):
    """What a provider returns for one call."""

    model_config = ConfigDict(frozen=True)

    text: str
    data: Any = None
    """Parsed JSON when the request carried a schema."""
    usage: Usage = Usage()
    model: str = ""
    """The model that actually produced the answer (may differ after a fallback)."""
    stop_reason: str | None = None


@runtime_checkable
class LLM(Protocol):
    """Anything that can answer an :class:`LLMRequest`."""

    @property
    def name(self) -> str:
        """Stable identifier such as ``anthropic:claude-opus-5``; recorded in results."""
        ...

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run one call.

        Raise :class:`~widethink.errors.RefusalError` or
        :class:`~widethink.errors.TruncatedOutputError` (with usage attached)
        when the output is unusable; let transport and authentication errors
        propagate unchanged.
        """
        ...
