"""Embeddings through the OpenAI API or any OpenAI-compatible server."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from widethink.embeddings.base import Embeddings, Vector, normalize
from widethink.errors import ConfigurationError


class OpenAIEmbedder:
    """``text-embedding-3-*`` (or a compatible model served elsewhere).

    Default thresholds suit ``text-embedding-3-small``: rephrasings of one idea
    score about 0.85 and above, loosely related ideas 0.45 and below.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 128,
        related_threshold: float = 0.45,
        duplicate_threshold: float = 0.85,
    ) -> None:
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ConfigurationError(
                    "OpenAIEmbedder needs the 'openai' package: pip install 'widethink[openai]'"
                ) from error
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._client = client
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._related = related_threshold
        self._duplicate = duplicate_threshold

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    @property
    def related_threshold(self) -> float:
        return self._related

    @property
    def duplicate_threshold(self) -> float:
        return self._duplicate

    async def embed(self, texts: Sequence[str]) -> Embeddings:
        vectors: list[Vector] = []
        tokens = 0
        for start in range(0, len(texts), self.batch_size):
            batch = [text or " " for text in texts[start : start + self.batch_size]]
            extra: dict[str, Any] = {"dimensions": self.dimensions} if self.dimensions else {}
            response = await self._client.embeddings.create(model=self.model, input=batch, **extra)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(normalize(item.embedding) for item in ordered)
            usage = getattr(response, "usage", None)
            tokens += int(getattr(usage, "total_tokens", 0) or 0)
        return Embeddings(vectors=vectors, tokens=tokens)
