from typing import Any

from app.config import settings
from app.embeddings.base import Embedder
from app.hybrid.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.models import SearchResult
from app.sparse.bm25_index import BM25Index
from app.vectorstore.base import VectorStore


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) search. Both branches apply
    tenant/ACL/metadata filters BEFORE truncating to their own candidate
    pool, so fusion never sees (and can never accidentally rank in) a
    result the caller wasn't allowed to see in the first place.
    """

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index, embedder: Embedder):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        pool_size = candidate_pool_size or settings.candidate_pool_size

        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_vector, top_k=pool_size, filters=filters)
        sparse_results = self.bm25_index.search(query, top_k=pool_size, filters=filters)

        dense_scores = dict(dense_results)
        sparse_scores = dict(sparse_results)

        if settings.fusion_method == "weighted":
            fused = weighted_score_fusion(
                dense_results,
                sparse_results,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )
        else:
            fused = reciprocal_rank_fusion(dense_results, sparse_results, k=settings.rrf_k)

        ranked_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)[:pool_size]

        results = []
        for chunk_id in ranked_ids:
            record = self.vector_store.get(chunk_id)
            if record is None:
                continue  # defensive: shouldn't happen if store/index are built together
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=record["document_id"],
                    text=record["text"],
                    metadata=record["metadata"],
                    dense_score=dense_scores.get(chunk_id),
                    sparse_score=sparse_scores.get(chunk_id),
                    fused_score=fused[chunk_id],
                )
            )
        # Returns the fused candidate pool (up to pool_size); callers that
        # don't rerank should slice [:top_k] themselves.
        return results
