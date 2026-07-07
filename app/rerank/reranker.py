import re
from abc import ABC, abstractmethod

from app.models import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return results sorted by rerank_score descending, with
        rerank_score populated on each SearchResult."""


class LexicalOverlapReranker(Reranker):
    """
    Offline fallback reranker: scores each candidate by query/chunk token
    overlap (Jaccard-style), independent of the fusion score already
    computed. It's intentionally cheap since it only runs over the small
    fused candidate pool, not the whole corpus.

    Swap in a real cross-encoder (e.g. a sentence-transformers
    CrossEncoder or a hosted rerank API) by implementing the same
    Reranker interface — the rest of the pipeline is unaffected.
    """

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_tokens = _tokens(query)
        if not query_tokens:
            for r in results:
                r.rerank_score = r.fused_score or 0.0
            return sorted(results, key=lambda r: r.rerank_score, reverse=True)

        for result in results:
            chunk_tokens = _tokens(result.text)
            overlap = len(query_tokens & chunk_tokens)
            union = len(query_tokens | chunk_tokens) or 1
            result.rerank_score = overlap / union

        return sorted(results, key=lambda r: r.rerank_score, reverse=True)
