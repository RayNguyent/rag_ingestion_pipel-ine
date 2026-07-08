from app.models import TenantContext
from app.retrieval.loader import load_engine
from scripts.build_index import build_index

# NOTE: previously this called
#   split_text(doc.text, chunk_size=50, chunk_overlap=10)
# which didn't match chunker.split_text's real signature
# (splitter_name, chunk_size, overlap) and would have raised a TypeError.
# Ingestion + chunking + embedding + indexing now all happen in
# scripts/build_index.py; this file demonstrates the resulting
# access-scoped retrieval engine.

if __name__ == "__main__":
    build_index()

    engine = load_engine()

    acme_ctx = TenantContext(tenant_id="acme", user_id="demo-user", roles=["*"])
    print("\n--- acme query: 'chunking strategy for retrieval' ---")
    for r in engine.retrieve("chunking strategy for retrieval", acme_ctx, top_k=3):
        print(f"  [{r.metadata['tenant_id']}] score={r.rerank_score:.3f}  {r.text[:70]}...")

    globex_ctx = TenantContext(tenant_id="globex", user_id="demo-user", roles=["employee"])
    print("\n--- globex query: 'expense report deadline' ---")
    for r in engine.retrieve("expense report deadline", globex_ctx, top_k=3):
        print(f"  [{r.metadata['tenant_id']}] score={r.rerank_score:.3f}  {r.text[:70]}...")

    print("\n--- globex caller querying acme-only vocabulary (should return 0 acme hits) ---")
    for r in engine.retrieve("tiktoken chunk overlap encoder", globex_ctx, top_k=3):
        print(f"  [{r.metadata['tenant_id']}] score={r.rerank_score:.3f}  {r.text[:70]}...")
