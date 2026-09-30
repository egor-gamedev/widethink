"""Embedders: turning ideas into vectors to measure repetition and relevance."""

from widethink.embeddings.base import Embedder, Embeddings, Vector, normalize
from widethink.embeddings.hashing import HashingEmbedder
from widethink.embeddings.openai import OpenAIEmbedder
from widethink.embeddings.sentence_transformers import SentenceTransformerEmbedder

__all__ = [
    "Embedder",
    "Embeddings",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "Vector",
    "normalize",
]
