"""Local embeddings with sentence-transformers (``pip install 'widethink[local]'``)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from widethink.embeddings.base import Embeddings, normalize
from widethink.errors import ConfigurationError


class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model on the local machine.

    The model loads lazily on first use; encoding runs in a worker thread so the
    event loop keeps serving concurrent model calls.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        related_threshold: float = 0.45,
        duplicate_threshold: float = 0.85,
    ) -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None
        self._related = related_threshold
        self._duplicate = duplicate_threshold

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    @property
    def related_threshold(self) -> float:
        return self._related

    @property
    def duplicate_threshold(self) -> float:
        return self._duplicate

    async def embed(self, texts: Sequence[str]) -> Embeddings:
        return await asyncio.to_thread(self._encode, list(texts))

    def _encode(self, texts: list[str]) -> Embeddings:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ConfigurationError(
                    "SentenceTransformerEmbedder needs 'sentence-transformers': "
                    "pip install 'widethink[local]'"
                ) from error
            self._model = SentenceTransformer(self.model_name, device=self.device)
        matrix = self._model.encode(texts, normalize_embeddings=True)
        return Embeddings(vectors=[normalize(row) for row in matrix])
