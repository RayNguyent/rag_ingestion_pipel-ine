import json
import sys

from app.evaluation.dataset import load_eval_set
from app.evaluation.evaluate import evaluate
from app.retrieval.loader import load_engine


def main(eval_set_path: str = "data/eval_set.json", k: int = 5) -> None:
    engine = load_engine()
    eval_set = load_eval_set(eval_set_path)
    summary = evaluate(engine, eval_set, k=k)

    per_query = summary.pop("per_query")
    print(json.dumps(summary, indent=2))
    print("\nPer-query breakdown:")
    for q in per_query:
        print(
            f"  [{q['tenant_id']}] '{q['query']}' -> "
            f"recall@k={q['recall@k']:.2f} mrr={q['mrr']:.2f} "
            f"ndcg@k={q['ndcg@k']:.2f} leaked={q['leaked_chunks']}"
        )

    if summary["total_cross_tenant_leaks"] > 0:
        print("\n[run_eval] WARNING: cross-tenant leakage detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
