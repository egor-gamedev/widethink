"""Calling a model with bounded retries: for a validated Pydantic object, or for plain text."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from widethink.errors import StructuredOutputError, TransientLLMError, TruncatedOutputError
from widethink.llm.base import LLM, LLMRequest, LLMResponse, Message, Usage
from widethink.llm.jsonschema import extract_json, strict_json_schema

T = TypeVar("T", bound=BaseModel)

#: Hard ceiling for retries after truncation; keeps non-streaming calls safe.
MAX_TOKENS_CEILING = 32_000

#: Called after every attempt: ``(usage, model, ok)``. Failed attempts count too.
AttemptHook = Callable[[Usage, str, bool], None]


async def generate_structured(
    llm: LLM,
    request: LLMRequest,
    output: type[T],
    *,
    retries: int = 1,
    on_attempt: AttemptHook | None = None,
) -> tuple[T, LLMResponse]:
    """Ask ``llm`` for an instance of ``output``.

    Truncated replies are retried with a doubled ``max_tokens``; transient
    provider failures are retried as they were; replies that do not validate are
    retried with the validation error shown to the model. Refusals and transport
    errors are not retried here.

    Raises:
        StructuredOutputError: if no attempt produced a valid object.
        TruncatedOutputError: if the last attempt was still truncated.
        TransientLLMError: if the last attempt failed on the provider's side.
    """
    current = request.model_copy(
        update={"json_schema": strict_json_schema(output), "schema_name": output.__name__}
    )
    last_error: StructuredOutputError | TruncatedOutputError | TransientLLMError | None = None
    for _ in range(retries + 1):
        try:
            response = await llm.generate(current)
        except TruncatedOutputError as error:
            _report(on_attempt, error.usage, error.model, ok=False)
            last_error = error
            current = _more_room(current)
            continue
        except TransientLLMError as error:
            _report(on_attempt, error.usage, error.model, ok=False)
            last_error = error
            continue
        try:
            payload = response.data if response.data is not None else extract_json(response.text)
            parsed = output.model_validate(payload)
        except (ValueError, ValidationError) as error:
            _report(on_attempt, response.usage, response.model, ok=False)
            last_error = StructuredOutputError(
                f"{output.__name__}: {error}", usage=response.usage, model=response.model
            )
            current = current.model_copy(
                update={"messages": (*request.messages, *_correction(response.text, error))}
            )
            continue
        _report(on_attempt, response.usage, response.model, ok=True)
        return parsed, response
    assert last_error is not None
    raise last_error


async def generate_text(
    llm: LLM,
    request: LLMRequest,
    *,
    retries: int = 1,
    on_attempt: AttemptHook | None = None,
) -> LLMResponse:
    """Ask ``llm`` for free text, retrying truncated and transiently failed calls.

    Raises:
        TruncatedOutputError: if the last attempt was still truncated.
        TransientLLMError: if the last attempt failed on the provider's side.
    """
    current = request
    last_error: TruncatedOutputError | TransientLLMError | None = None
    for _ in range(retries + 1):
        try:
            response = await llm.generate(current)
        except TruncatedOutputError as error:
            _report(on_attempt, error.usage, error.model, ok=False)
            last_error = error
            current = _more_room(current)
            continue
        except TransientLLMError as error:
            _report(on_attempt, error.usage, error.model, ok=False)
            last_error = error
            continue
        _report(on_attempt, response.usage, response.model, ok=True)
        return response
    assert last_error is not None
    raise last_error


def _more_room(request: LLMRequest) -> LLMRequest:
    return request.model_copy(
        update={"max_tokens": min(request.max_tokens * 2, MAX_TOKENS_CEILING)}
    )


def _correction(previous: str, error: Exception) -> tuple[Message, Message]:
    detail = str(error)[:1500]
    return (
        Message(role="assistant", content=previous or "(empty reply)"),
        Message(
            role="user",
            content=(
                "Your previous reply did not match the required JSON schema:\n"
                f"{detail}\n\nReply again with only the JSON object."
            ),
        ),
    )


def _report(hook: AttemptHook | None, usage: Usage | None, model: str, *, ok: bool) -> None:
    if hook is not None:
        hook(usage or Usage(), model, ok)
