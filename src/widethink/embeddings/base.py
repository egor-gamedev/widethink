"""Embedders measure how close two ideas are.

Closeness drives three mechanisms: satiation (a branch repeating what the tree
already holds gets boring), inhibition of return (duplicates of considered
ideas are pruned) and reinstatement (the context items nearest to a thought
accompany it). Each embedder states its own similarity thresholds, because
cosine values are not comparable between embedding models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Embeddings:
    """Vectors for a batch of texts, L2-normalized, plus the tokens billed for them."""

    vectors: list[Vector]
    tokens: int = 0


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into normalized vectors."""

    @property
    def name(self) -> str: ...

    @property
    def related_threshold(self) -> float:
        """Cosine similarity at or below which two ideas count as different."""
        ...

    @property
    def duplicate_threshold(self) -> float:
        """Cosine similarity at or above which two ideas count as the same."""
        ...

    async def embed(self, texts: Sequence[str]) -> Embeddings: ...


def normalize(values: npt.ArrayLike) -> Vector:
    """Return ``values`` as a float32 unit vector (a zero vector stays zero)."""
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return (vector / norm).astype(np.float32)
