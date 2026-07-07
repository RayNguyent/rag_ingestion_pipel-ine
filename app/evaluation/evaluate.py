from app.evaluation.dataset import EvalExample
from app.evaluation.metrics import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from app.retrieval.engine import ContextRetrievalEngine


def evaluate(
    engine: ContextRetrievalEngine,
    eval_set: list[EvalExample],
    k: int = 5,
) -> dict:
    per_query = []
    leakage_count = 0

    for example in eval_set:
        tenant_context = example.tenant_context()
        results = engine.retrieve(example.query, tenant_context, top_k=k)
        relevant_ids = set(example.relevant_chunk_ids)

        # Cross-tenant leakage check: every returned chunk's tenant_id MUST
        # match the caller's tenant. This is evaluated independently of the
        # access guard in app/access/guard.py, as an end-to-end sanity check.
        leaked = [r for r in results if r.metadata.get("tenant_id") != example.tenant_id]
        leakage_count += len(leaked)

        per_query.append(
            {
                "query": example.query,
                "tenant_id": example.tenant_id,
                "recall@k": recall_at_k(results, relevant_ids, k),
                "precision@k": precision_at_k(results, relevant_ids, k),
                "mrr": mrr(results, relevant_ids),
                "ndcg@k": ndcg_at_k(results, relevant_ids, k),
                "hit_rate@k": hit_rate(results, relevant_ids, k),
                "leaked_chunks": len(leaked),
            }
        )

    n = len(per_query) or 1
    summary = {
        "num_queries": len(per_query),
        "avg_recall@k": sum(q["recall@k"] for q in per_query) / n,
        "avg_precision@k": sum(q["precision@k"] for q in per_query) / n,
        "avg_mrr": sum(q["mrr"] for q in per_query) / n,
        "avg_ndcg@k": sum(q["ndcg@k"] for q in per_query) / n,
        "avg_hit_rate@k": sum(q["hit_rate@k"] for q in per_query) / n,
        "total_cross_tenant_leaks": leakage_count,
        "per_query": per_query,
    }
    return summary
