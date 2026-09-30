"""Record model calls to JSONL and replay them for exact, offline reproduction.

A research result is only as good as its reproducibility. Wrapping a provider
in :class:`RecordingLLM` stores every request and response; :class:`ReplayLLM`
later answers the same requests from the file without network access or cost.
Requests are matched by a hash of their full content, so a replay also detects
any change in prompts or in the harness's decisions (``ReplayMismatchError``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from widethink.errors import (
    LLMError,
    RefusalError,
    ReplayMismatchError,
    StructuredOutputError,
    TruncatedOutputError,
)
from widethink.llm.base import LLM, LLMRequest, LLMResponse, Usage

_ERRORS: dict[str, type[LLMError]] = {
    cls.__name__: cls
    for cls in (LLMError, RefusalError, TruncatedOutputError, StructuredOutputError)
}


def request_key(request: LLMRequest) -> str:
    """Content hash identifying a request."""
    payload = json.dumps(
        request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RecordingLLM:
    """Pass calls through to ``inner`` and append each exchange to a JSONL file."""

    def __init__(self, inner: LLM, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self.inner.name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self.inner.generate(request)
        except LLMError as error:
            await self._write(request, response=None, error=error)
            raise
        await self._write(request, response=response, error=None)
        return response

    async def _write(
        self, request: LLMRequest, *, response: LLMResponse | None, error: LLMError | None
    ) -> None:
        record: dict[str, Any] = {
            "key": request_key(request),
            "provider": self.inner.name,
            "request": request.model_dump(mode="json"),
            "response": None if response is None else response.model_dump(mode="json"),
            "error": None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error),
                "model": error.model,
                "usage": None if error.usage is None else error.usage.model_dump(),
            },
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)


class ReplayLLM:
    """Answer requests from a recording made by :class:`RecordingLLM`."""

    def __init__(self, path: str | Path, *, name: str | None = None) -> None:
        self._entries: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        provider = None
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    provider = provider or entry.get("provider")
                    self._entries[entry["key"]].append(entry)
        self._name = name or provider or "replay"

    @property
    def name(self) -> str:
        return self._name

    def remaining(self) -> int:
        """Recorded exchanges not yet replayed."""
        return sum(len(queue) for queue in self._entries.values())

    async def generate(self, request: LLMRequest) -> LLMResponse:
        key = request_key(request)
        queue = self._entries.get(key)
        if not queue:
            raise ReplayMismatchError(
                f"no recorded response for this {request.purpose!r} request (key {key[:12]}); "
                "prompts, configuration or the harness's decisions differ from the recording"
            )
        entry = queue.popleft()
        await asyncio.sleep(0)
        if entry["error"] is not None:
            error = entry["error"]
            usage = None if error["usage"] is None else Usage.model_validate(error["usage"])
            raise _ERRORS.get(error["type"], LLMError)(
                error["message"], usage=usage, model=error["model"]
            )
        return LLMResponse.model_validate(entry["response"])
