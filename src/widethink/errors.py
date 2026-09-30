"""Exception hierarchy.

Every error raised on purpose by widethink derives from :class:`WideThinkError`.
Errors of a model call carry the token usage of the failed attempt, because a
failed call still costs tokens and equal-budget comparisons must count it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from widethink.llm.base import Usage


class WideThinkError(Exception):
    """Base class for all widethink errors."""


class ConfigurationError(WideThinkError):
    """Invalid configuration or a missing optional dependency."""


class LLMError(WideThinkError):
    """A model call produced no usable output."""

    def __init__(self, message: str, *, usage: Usage | None = None, model: str = "") -> None:
        super().__init__(message)
        self.usage = usage
        self.model = model


class RefusalError(LLMError):
    """The model declined the request (for example, a safety refusal)."""


class TruncatedOutputError(LLMError):
    """The output hit ``max_tokens`` before it was complete."""


class StructuredOutputError(LLMError):
    """The output could not be parsed or validated against the expected schema."""


class TransientLLMError(LLMError):
    """The provider could not finish the output this time (overload, abort); safe to retry."""


class ReplayMismatchError(WideThinkError):
    """A replayed run issued a request that is not present in the recording."""
