from app.evaluation.metrics import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from app.models import SearchResult


def _result(chunk_id: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, document_id="d1", text="x", metadata={})


def test_recall_and_precision_perfect_match():
    results = [_result("a"), _result("b"), _result("c")]
    relevant = {"a", "b"}

    assert recall_at_k(results, relevant, k=3) == 1.0
    assert precision_at_k(results, relevant, k=3) == 2 / 3


def test_mrr_finds_first_relevant_rank():
    results = [_result("x"), _result("a"), _result("y")]
    relevant = {"a"}

    assert mrr(results, relevant) == 1 / 2


def test_ndcg_perfect_ranking_is_one():
    results = [_result("a"), _result("b")]
    relevant = {"a", "b"}

    assert ndcg_at_k(results, relevant, k=2) == 1.0


def test_hit_rate_zero_when_nothing_relevant_returned():
    results = [_result("x"), _result("y")]
    relevant = {"a"}

    assert hit_rate(results, relevant, k=2) == 0.0


def test_metrics_handle_empty_relevant_set_gracefully():
    results = [_result("x")]
    assert recall_at_k(results, set(), k=1) == 0.0
