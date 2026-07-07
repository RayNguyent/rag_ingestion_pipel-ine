from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VectorStore(ABC):
    """
    Interface every vector backend must implement. The bundled NumpyVectorStore
    is a fully offline, dependency-light default. For production scale, swap
    in Qdrant/Chroma/pgvector by implementing this same interface — the
    filter semantics (see app/filters.py) should be preserved so tenant/ACL
    enforcement behaves identically regardless of backend.
    """

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict],
        texts: list[str],
        document_ids: list[str],
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Returns [(chunk_id, similarity_score), ...] sorted descending.
        Filtering MUST happen before top_k truncation, never after."""

    @abstractmethod
    def get(self, chunk_id: str) -> dict | None:
        """Fetch a stored record (text + metadata + document_id) by id."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
