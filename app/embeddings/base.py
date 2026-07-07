from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """
    Interface every embedding backend must implement. Swap the default
    TF-IDF/LSA backend for sentence-transformers, OpenAI, Cohere, etc. by
    implementing this same interface — nothing else in the pipeline needs
    to change.
    """

    dim: int

    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Fit the embedder on the full corpus (no-op for API-based embedders)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents/chunks. Returns shape (n, dim), L2-normalized."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,), L2-normalized."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
