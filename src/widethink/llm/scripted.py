"""A deterministic stand-in for a model: for tests, examples and dry runs."""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

from pydantic import BaseModel

from widethink.llm.base import LLMRequest, LLMResponse, Usage
from widethink.llm.jsonschema import extract_json

Reply: TypeAlias = "str | Mapping[str, Any] | BaseModel | Exception"
Responder: TypeAlias = Callable[[LLMRequest], Reply]


def estimate_tokens(text: str) -> int:
    """Rough token count (about four characters per token)."""
    return math.ceil(len(text) / 4) if text else 0


class ScriptedLLM:
    """Answers from a function or a fixed queue of replies.

    A reply may be a string, a mapping or a Pydantic model (serialized to JSON),
    or an exception instance, which is raised - handy for testing failure paths.
    Usage is estimated from text length so that budgets behave realistically.
    All requests are kept in :attr:`requests` for assertions.
    """

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        replies: Iterable[Reply] = (),
        name: str = "scripted",
    ) -> None:
        self._responder = responder
        self._replies: deque[Reply] = deque(replies)
        if responder is None and not self._replies:
            raise ValueError("give a responder or a sequence of replies")
        self._name = name
        self.requests: list[LLMRequest] = []

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        await asyncio.sleep(0)  # yield like a real network call would
        if self._responder is not None:
            reply = self._responder(request)
        elif self._replies:
            reply = self._replies.popleft()
        else:
            raise RuntimeError("ScriptedLLM ran out of replies")
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, BaseModel):
            text = reply.model_dump_json()
        elif isinstance(reply, str):
            text = reply
        else:
            text = json.dumps(dict(reply), ensure_ascii=False)
        data: Any = None
        if request.json_schema is not None:
            try:
                data = extract_json(text)
            except ValueError:
                data = None
        prompt = request.system + "".join(message.content for message in request.messages)
        return LLMResponse(
            text=text,
            data=data,
            usage=Usage(input_tokens=estimate_tokens(prompt), output_tokens=estimate_tokens(text)),
            model=self._name,
            stop_reason="end_turn",
        )
