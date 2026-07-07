from typing import Any

from app.access.guard import enforce
from app.access.policy import build_filters
from app.config import settings
from app.hybrid.retriever import HybridRetriever
from app.models import SearchResult, TenantContext
from app.rerank.reranker import Reranker


class ContextRetrievalEngine:
    """
    The single public entrypoint for retrieval. Every caller — CLI, API
    layer, evaluation harness — should go through this class rather than
    touching HybridRetriever/VectorStore/BM25Index directly, so tenant/ACL
    enforcement can never accidentally be skipped.

    Pipeline: build tenant-scoped filters -> hybrid dense+sparse retrieve
    (already filtered pre-fusion) -> access guard (defense in depth) ->
    rerank -> truncate to top_k.
    """

    def __init__(self, hybrid_retriever: HybridRetriever, reranker: Reranker):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        tenant_context: TenantContext,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or settings.default_top_k

        scoped_filters = build_filters(tenant_context, filters)

        candidates = self.hybrid_retriever.retrieve(
            query=query,
            filters=scoped_filters,
            candidate_pool_size=candidate_pool_size,
        )

        safe_candidates = enforce(candidates, tenant_context, strict=settings.enforce_tenant_isolation)

        reranked = self.reranker.rerank(query, safe_candidates)

        return reranked[:top_k]
