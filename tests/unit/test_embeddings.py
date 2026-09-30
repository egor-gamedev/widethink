from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from widethink.embeddings import Embeddings, HashingEmbedder, normalize
from widethink.semantic import SemanticIndex


def cosine(embedder: HashingEmbedder, a: str, b: str) -> float:
    return float(embedder.vector(a) @ embedder.vector(b))


class TestHashingEmbedder:
    def test_vectors_are_unit_length_and_deterministic(self) -> None:
        first = HashingEmbedder().vector("Refresh token rotation")
        second = HashingEmbedder().vector("Refresh token rotation")
        assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
        assert np.array_equal(first, second)

    def test_rephrasing_is_closer_than_an_unrelated_idea(self) -> None:
        embedder = HashingEmbedder()
        same = cosine(embedder, "Rotate refresh tokens on every use", "Refresh token rotation")
        other = cosine(embedder, "Rotate refresh tokens on every use", "Technicians work offline")
        assert same > embedder.related_threshold > other

    def test_near_identical_labels_are_duplicates(self) -> None:
        embedder = HashingEmbedder()
        score = cosine(embedder, "Device-bound tokens", "Device bound tokens.")
        assert score >= embedder.duplicate_threshold

    def test_inflections_share_features_across_languages(self) -> None:
        embedder = HashingEmbedder()
        assert cosine(embedder, "токены устройства", "токен устройств") > 0.5

    def test_stopwords_do_not_make_texts_similar(self) -> None:
        embedder = HashingEmbedder()
        assert cosine(embedder, "the and of for", "and the of for") == 0.0

    def test_small_dimension_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            HashingEmbedder(dim=8)

    async def test_async_interface(self) -> None:
        batch = await HashingEmbedder().embed(["a b", "c d"])
        assert len(batch.vectors) == 2
        assert batch.tokens == 0


def test_normalize_keeps_zero_vectors() -> None:
    assert not normalize([0.0, 0.0]).any()
    assert normalize([3.0, 4.0]).tolist() == pytest.approx([0.6, 0.8])


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> Embeddings:
        self.batches.append(list(texts))
        batch = await super().embed(texts)
        return Embeddings(vectors=batch.vectors, tokens=len(texts))


class TestSemanticIndex:
    async def test_embeddings_are_cached_by_text(self) -> None:
        embedder = CountingEmbedder()
        spent: list[int] = []
        index = SemanticIndex(embedder, on_tokens=spent.append)
        await index.embed(["a", "b", "a"])
        await index.embed(["b", "c"])
        assert embedder.batches == [["a", "b"], ["c"]]
        assert spent == [2, 1]

    async def test_nearest_and_rank(self) -> None:
        index = SemanticIndex(HashingEmbedder())
        texts = ["offline technicians", "offline field technicians", "signing keys"]
        for key, vector in zip("abc", await index.embed(texts), strict=True):
            index.put(key, vector)
        query = index.get("a")
        assert query is not None
        assert index.nearest(query, ["b", "c"])[1] == "b"
        assert [key for key, _ in index.rank(query, ["c", "b", "missing"])] == ["b", "c"]
        assert index.nearest(query, []) == (0.0, None)
        assert index.rank(query, ["missing"]) == []
        assert "a" in index
        assert index.related < index.duplicate

    async def test_wrong_vector_count_is_an_error(self) -> None:
        class Broken(HashingEmbedder):
            async def embed(self, texts: Sequence[str]) -> Embeddings:
                return Embeddings(vectors=[])

        with pytest.raises(ValueError, match="wrong number"):
            await SemanticIndex(Broken()).embed(["x"])
