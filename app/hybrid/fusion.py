def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """
    RRF: fused_score(id) = sum(1 / (k + rank)) across every ranked list the
    id appears in. Rank-based, so it needs no score normalization between
    dense cosine similarity and BM25 scores (which live on different scales).
    """
    fused: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense_results):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(sparse_results):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


def weighted_score_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> dict[str, float]:
    """
    Weighted sum of min-max normalized scores. More tunable than RRF but
    sensitive to score distribution — prefer RRF unless you've calibrated
    these weights on real query traffic.
    """

    def _normalize(results: list[tuple[str, float]]) -> dict[str, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {cid: (s - lo) / span for cid, s in results}

    dense_norm = _normalize(dense_results)
    sparse_norm = _normalize(sparse_results)

    fused: dict[str, float] = {}
    for chunk_id in set(dense_norm) | set(sparse_norm):
        fused[chunk_id] = (
            dense_weight * dense_norm.get(chunk_id, 0.0)
            + sparse_weight * sparse_norm.get(chunk_id, 0.0)
        )
    return fused
