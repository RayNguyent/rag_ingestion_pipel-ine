import pickle
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.filters import matches_filters

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]: # break down query or document text into tokens (words) for BM25 indexing and searching
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Sparse keyword index (BM25). Kept as a standalone module so it can be
    built, persisted, and queried independently of the dense vector store,
    and combined later in hybrid/retriever.py.
    """

    def __init__(self):
        self.ids: list[str] = []
        self.metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def build(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        self.ids = ids
        self.metadatas = metadatas
        self._tokenized_corpus = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.ids:
            return []

        scores = self._bm25.get_scores(tokenize(query))

        # Filter FIRST, then truncate — identical policy to NumpyVectorStore.
        allowed = [i for i, meta in enumerate(self.metadatas) if matches_filters(meta, filters)]
        if not allowed:
            return []

        allowed_scores = [(i, scores[i]) for i in allowed]
        allowed_scores.sort(key=lambda x: x[1], reverse=True) # sort by score in descending order
        top = allowed_scores[:top_k]
        return [(self.ids[i], float(score)) for i, score in top]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "metadatas": self.metadatas,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.ids = state["ids"]
        self.metadatas = state["metadatas"]
        self._tokenized_corpus = state["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
