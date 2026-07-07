from app.hybrid.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.models import TenantContext


def test_hybrid_retrieve_returns_relevant_results(two_tenant_engine):
    engine, _, _ = two_tenant_engine
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])

    results = engine.retrieve("cloud infrastructure margin", ctx, top_k=3)

    assert len(results) > 0
    assert results[0].fused_score is not None
    assert results[0].rerank_score is not None
    # reranked order should be non-increasing by rerank_score
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reciprocal_rank_fusion_combines_both_lists():
    dense = [("a", 0.9), ("b", 0.5)]
    sparse = [("b", 10.0), ("c", 8.0)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    assert set(fused.keys()) == {"a", "b", "c"}
    # "b" appears in both lists, so it should score higher than any id
    # appearing in only one list.
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_weighted_score_fusion_respects_weights():
    dense = [("a", 1.0), ("b", 0.0)]
    sparse = [("a", 0.0), ("b", 1.0)]

    dense_heavy = weighted_score_fusion(dense, sparse, dense_weight=1.0, sparse_weight=0.0)
    sparse_heavy = weighted_score_fusion(dense, sparse, dense_weight=0.0, sparse_weight=1.0)

    assert dense_heavy["a"] > dense_heavy["b"]
    assert sparse_heavy["b"] > sparse_heavy["a"]


def test_filters_applied_before_truncation_do_not_starve_results(two_tenant_engine):
    """Even with a small candidate_pool_size, filtering happens before
    truncation, so an in-tenant query still returns results rather than
    being crowded out by the other tenant's higher-scoring chunks."""
    engine, _, _ = two_tenant_engine
    ctx = TenantContext(tenant_id="tenant-b", user_id="u1", roles=["*"])

    results = engine.retrieve(
        "training policy orientation", ctx, top_k=3, candidate_pool_size=2
    )

    assert len(results) > 0
    assert all(r.metadata["tenant_id"] == "tenant-b" for r in results)
