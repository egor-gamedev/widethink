"""A dependency-free embedder based on feature hashing.

It needs no model and no network, so tests and offline demos are fast and
fully reproducible. It sees surface form, not meaning: word and word-pair
features catch rephrasings that reuse words, character n-grams catch
inflections ("token" / "tokens", "токен" / "токены"). Use a neural embedder for
real runs; this one is a fallback.

Hashing uses BLAKE2b rather than Python's ``hash``, which is salted per
process and would make vectors differ between runs.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from widethink.embeddings.base import Embeddings, Vector, normalize

_WORD = re.compile(r"\w+", re.UNICODE)

_STOPWORDS_TEXT = """
a an and are as at be by for from has have in is it its of on or that the this to was
were will with we you your our not no do does can should would could may might must
и в во на с со к ко по о об от до за из у не но а же ли бы то это как что чтобы
для при без над под или его её их мы вы они он она оно
"""
#: Frequent function words (English and Russian) carry no meaning but inflate similarity.
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

#: Calibrated on labelled pairs (duplicates, paraphrases, unrelated ideas; English and
#: Russian): duplicates score >= 0.84, paraphrases 0.55-0.79, unrelated ideas <= 0.04.
_WEIGHTS = {"word": 1.0, "pair": 0.3, "gram": 1.0}


class HashingEmbedder:
    """Signed feature hashing of words, word pairs and character n-grams."""

    def __init__(
        self,
        dim: int = 4096,
        *,
        ngram: int = 3,
        related_threshold: float = 0.2,
        duplicate_threshold: float = 0.8,
    ) -> None:
        if dim < 64:
            raise ValueError("dim must be at least 64")
        self.dim = dim
        self.ngram = ngram
        self._related = related_threshold
        self._duplicate = duplicate_threshold

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    @property
    def related_threshold(self) -> float:
        return self._related

    @property
    def duplicate_threshold(self) -> float:
        return self._duplicate

    async def embed(self, texts: Sequence[str]) -> Embeddings:
        return Embeddings(vectors=[self.vector(text) for text in texts])

    def vector(self, text: str) -> Vector:
        """Embed one text synchronously."""
        values = np.zeros(self.dim, dtype=np.float32)
        for feature, weight in self._features(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            sign = 1.0 if number >> 63 else -1.0
            values[number % self.dim] += sign * weight
        return normalize(values)

    def _features(self, text: str) -> dict[str, float]:
        words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
        counts: Counter[str] = Counter()
        for word in words:
            counts[f"w:{word}"] += 1
            padded = f"<{word}>"
            for start in range(max(1, len(padded) - self.ngram + 1)):
                counts[f"g:{padded[start : start + self.ngram]}"] += 1
        for left, right in itertools.pairwise(words):
            counts[f"p:{left} {right}"] += 1
        features: dict[str, float] = {}
        for feature, count in counts.items():
            kind = {"w": "word", "p": "pair", "g": "gram"}[feature[0]]
            features[feature] = _WEIGHTS[kind] * (1.0 + math.log(count))
        return features
