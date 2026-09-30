"""Embedders and provider constructors that talk to external services, with fakes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from widethink.embeddings import OpenAIEmbedder, SentenceTransformerEmbedder
from widethink.llm import AnthropicLLM, OpenAICompatibleLLM


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        batch = kwargs["input"]
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0, 0.0])
            for i, text in enumerate(batch)
        ]
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(total_tokens=5))


async def test_openai_embedder_orders_normalizes_batches_and_counts_tokens() -> None:
    embeddings = FakeEmbeddings()
    embedder = OpenAIEmbedder(
        client=SimpleNamespace(embeddings=embeddings), dimensions=3, batch_size=2
    )
    batch = await embedder.embed(["ab", "", "abcd"])
    assert len(embeddings.calls) == 2
    assert embeddings.calls[0]["input"] == ["ab", " "]  # empty text is not sent as-is
    assert embeddings.calls[0]["dimensions"] == 3
    assert embeddings.calls[0]["model"] == "text-embedding-3-small"
    first = batch.vectors[0]
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert first[0] > batch.vectors[1][0]  # order follows the input, not the response
    assert batch.tokens == 10
    assert embedder.name == "openai:text-embedding-3-small"
    assert embedder.related_threshold < embedder.duplicate_threshold


class FakeSentenceModel:
    def encode(self, texts: list[str], normalize_embeddings: bool) -> Any:
        assert normalize_embeddings
        return np.array([[3.0, 4.0] for _ in texts])


async def test_sentence_transformer_embedder_uses_the_loaded_model() -> None:
    embedder = SentenceTransformerEmbedder("tiny")
    embedder._model = FakeSentenceModel()
    batch = await embedder.embed(["a", "b"])
    assert [v.tolist() for v in batch.vectors] == [pytest.approx([0.6, 0.8])] * 2
    assert embedder.name == "st:tiny"
    assert embedder.related_threshold < embedder.duplicate_threshold


def test_provider_constructors_build_sdk_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    claude = AnthropicLLM(max_retries=1, timeout=5.0)
    assert claude._client.max_retries == 1
    local = OpenAICompatibleLLM("llama", base_url="http://localhost:8000/v1")
    assert str(local._client.base_url).startswith("http://localhost:8000")
    embedder = OpenAIEmbedder(base_url="http://localhost:8000/v1")
    assert embedder.name == "openai:text-embedding-3-small"
