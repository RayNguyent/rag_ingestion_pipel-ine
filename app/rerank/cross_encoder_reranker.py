from sentence_transformers import CrossEncoder
from app.models import SearchResult
from app.rerank.reranker import Reranker


class CrossEncoderReranker(Reranker):
    """
    Production reranker using a sentence-transformers CrossEncoder.

    Typical models:
        cross-encoder/ms-marco-MiniLM-L-6-v2
        cross-encoder/ms-marco-MiniLM-L12-v2
        BAAI/bge-reranker-base
        BAAI/bge-reranker-large

    Scores are generated directly from
    (query, chunk) pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [
            (query, result.text)
            for result in results
        ]

        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        return sorted(
            results,
            key=lambda r: r.rerank_score,
            reverse=True,
        )