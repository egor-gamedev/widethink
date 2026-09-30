"""Vector memory of a run: which ideas (and context items) are close to which."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from widethink.embeddings.base import Embedder, Vector


class SemanticIndex:
    """Caches embeddings by text and answers nearest-neighbour questions by key."""

    def __init__(self, embedder: Embedder, *, on_tokens: Callable[[int], None] | None = None):
        self.embedder = embedder
        self._on_tokens = on_tokens
        self._by_text: dict[str, Vector] = {}
        self._by_key: dict[str, Vector] = {}

    @property
    def related(self) -> float:
        return self.embedder.related_threshold

    @property
    def duplicate(self) -> float:
        return self.embedder.duplicate_threshold

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed texts, calling the embedder only for texts not seen before."""
        missing = list(dict.fromkeys(text for text in texts if text not in self._by_text))
        if missing:
            batch = await self.embedder.embed(missing)
            if len(batch.vectors) != len(missing):
                raise ValueError("embedder returned a wrong number of vectors")
            self._by_text.update(zip(missing, batch.vectors, strict=True))
            if batch.tokens and self._on_tokens is not None:
                self._on_tokens(batch.tokens)
        return [self._by_text[text] for text in texts]

    def put(self, key: str, vector: Vector) -> None:
        self._by_key[key] = vector

    def get(self, key: str) -> Vector | None:
        return self._by_key.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    def nearest(self, vector: Vector, keys: Iterable[str]) -> tuple[float, str | None]:
        """Highest cosine similarity between ``vector`` and the vectors of ``keys``."""
        present = [key for key in keys if key in self._by_key]
        if not present:
            return 0.0, None
        matrix = np.stack([self._by_key[key] for key in present])
        similarities = matrix @ vector
        best = int(np.argmax(similarities))
        return float(similarities[best]), present[best]

    def rank(self, vector: Vector, keys: Iterable[str]) -> list[tuple[str, float]]:
        """``keys`` ordered by decreasing similarity to ``vector``."""
        present = [key for key in keys if key in self._by_key]
        if not present:
            return []
        similarities = np.stack([self._by_key[key] for key in present]) @ vector
        order = np.argsort(-similarities, kind="stable")
        return [(present[int(i)], float(similarities[int(i)])) for i in order]
