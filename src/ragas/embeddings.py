"""Embeddings interface backed by agno's OpenAIEmbedder."""

import typing as t

import numpy as np


class Embeddings:
    """Duck-typed interface: embed_query / embed_documents."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbeddings(Embeddings):
    """Wraps agno.knowledge.embedder.openai.OpenAIEmbedder."""

    def __init__(self, id: str = "text-embedding-3-small", **kwargs: t.Any):
        from agno.knowledge.embedder.openai import OpenAIEmbedder

        self._embedder = OpenAIEmbedder(id=id, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.get_embedding(t) for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between a (n, d) and b (m, d) -> (m,)."""
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
    return np.dot(a, b) / norm
