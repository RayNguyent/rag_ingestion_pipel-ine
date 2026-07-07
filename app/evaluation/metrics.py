import math

from app.models import SearchResult


def recall_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = {r.chunk_id for r in results[:k]}
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    retrieved = {r.chunk_id for r in top_k}
    return len(retrieved & relevant_ids) / len(top_k)


def mrr(results: list[SearchResult], relevant_ids: set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    top_k = results[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, result in enumerate(top_k, start=1)
        if result.chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    top_k_ids = {r.chunk_id for r in results[:k]}
    return 1.0 if top_k_ids & relevant_ids else 0.0
