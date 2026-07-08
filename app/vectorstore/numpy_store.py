import json
from pathlib import Path
from typing import Any

import numpy as np

from app.filters import matches_filters
from app.vectorstore.base import VectorStore


class NumpyVectorStore(VectorStore):
    """
    Minimal, dependency-light vector store: keeps L2-normalized vectors in a
    single numpy matrix and does brute-force cosine similarity. Fine for
    the dataset sizes this pipeline ingests locally, and requires no
    external server. Swap for Qdrant/Chroma/pgvector at scale by
    implementing the same VectorStore interface.
    """

    def __init__(self):
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None  # shape (n, dim)
        self.metadatas: list[dict] = []
        self.texts: list[str] = []
        self.document_ids: list[str] = []
        self._id_to_row: dict[str, int] = {}

    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict],
        texts: list[str],
        document_ids: list[str],
    ) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # avoids division by zero for zero vectors
        vectors = vectors / norms

        for i, chunk_id in enumerate(ids):
            if chunk_id in self._id_to_row:
                row = self._id_to_row[chunk_id]
                self.vectors[row] = vectors[i]
                self.metadatas[row] = metadatas[i]
                self.texts[row] = texts[i]
                self.document_ids[row] = document_ids[i]
            else:
                self._id_to_row[chunk_id] = len(self.ids)
                self.ids.append(chunk_id)
                self.metadatas.append(metadatas[i])
                self.texts.append(texts[i])
                self.document_ids.append(document_ids[i])
                self.vectors = (
                    vectors[i : i + 1]
                    if self.vectors is None
                    else np.vstack([self.vectors, vectors[i : i + 1]])
                )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if self.vectors is None or len(self.ids) == 0: # no vector - no result
            return [] 

        # 1. Apply metadata/tenant/ACL filters FIRST, before any truncation.
        allowed_rows = [
            row for row, meta in enumerate(self.metadatas) if matches_filters(meta, filters)
        ]
        if not allowed_rows:
            return []

        qv = query_vector / (np.linalg.norm(query_vector) or 1.0) # normalize query vector to unit length (avoid division by zero)
        candidate_vectors = self.vectors[allowed_rows]
        scores = candidate_vectors @ qv  # cosine similarity (vectors are normalized)

        # 2. Only now truncate to top_k, over the already-filtered candidate set.
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[allowed_rows[i]], float(scores[i])) for i in order]

    def get(self, chunk_id: str) -> dict | None:
        row = self._id_to_row.get(chunk_id)
        if row is None:
            return None
        return {
            "chunk_id": chunk_id,
            "document_id": self.document_ids[row],
            "text": self.texts[row],
            "metadata": self.metadatas[row],
        }

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(Path(path) / "vectors.npy", self.vectors)
        with open(Path(path) / "records.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": self.ids,
                    "metadatas": self.metadatas,
                    "texts": self.texts,
                    "document_ids": self.document_ids,
                },
                f,
            )

    def load(self, path: str) -> None:
        self.vectors = np.load(Path(path) / "vectors.npy")
        with open(Path(path) / "records.json", encoding="utf-8") as f:
            data = json.load(f)
        self.ids = data["ids"]
        self.metadatas = data["metadatas"]
        self.texts = data["texts"]
        self.document_ids = data["document_ids"]
        self._id_to_row = {cid: i for i, cid in enumerate(self.ids)}
