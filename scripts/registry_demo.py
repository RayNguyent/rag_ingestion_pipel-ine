"""
Demonstrates the tool registry end to end against the real project
components (not toy examples):

  1. build the index (scripts/build_index.py, unchanged)
  2. register_all() — wires retrieve / answer / ingest_document into the
     process-wide registry
  3. a READ-ONLY caller successfully calls "retrieve"
  4. the SAME caller is denied on "ingest_document" (no write scope)
  5. an INGESTION-scoped caller successfully calls "ingest_document" to add
     a brand-new document to the persisted index
  6. the newly ingested content is retrievable immediately afterwards
  7. print the accumulated monitoring metrics for every call made above

Run with: python -m scripts.registry_demo
"""

import json

from app.models import TenantContext
from app.registry import (
    CallerContext,
    PermissionDeniedError,
    guarded_execute,
    metrics,
    monitored_execute,
    registry,
)
from app.registry.access import ingestion_caller, read_only_caller
from app.registry.tools import register_all
from scripts.build_index import build_index


def main() -> None:
    build_index()
    register_all()

    print(f"\nread tools:  {[t.name for t in registry.read_tools()]}")
    print(f"write tools: {[t.name for t in registry.write_tools()]}")

    acme_tenant = TenantContext(tenant_id="acme", user_id="analyst-1", roles=["*"])
    reader = read_only_caller(acme_tenant)

    print("\n--- read-only caller: retrieve (should succeed) ---")
    result = monitored_execute(
        "retrieve",
        {"query": "chunking strategy for retrieval", "tenant_context": acme_tenant.model_dump(), "top_k": 3},
        reader,
    )
    for r in result.results:
        print(f"  score={r.rerank_score:.3f}  {r.text[:70]}...")

    print("\n--- read-only caller: ingest_document (should be DENIED) ---")
    try:
        monitored_execute(
            "ingest_document",
            {"path": "data/acme_deployment_notes.txt", "tenant_id": "acme"},
            reader,
        )
        print("  UNEXPECTED: write succeeded for a read-only caller")
    except PermissionDeniedError as e:
        print(f"  denied as expected: {e}")

    print("\n--- ingestion-scoped caller: ingest_document (should succeed) ---")
    admin_ctx = TenantContext(tenant_id="acme", user_id="admin-1", roles=["*"])
    writer = ingestion_caller(admin_ctx)
    ingest_result = monitored_execute(
        "ingest_document",
        {
            "path": "data/acme_deployment_notes.txt",
            "tenant_id": "acme",
            "extra_metadata": {"doc_type": "runbook"},
        },
        writer,
    )
    print(f"  ingested: {ingest_result.model_dump()}")

    print("\n--- read-only caller: retrieve newly ingested content (should find it) ---")
    result = monitored_execute(
        "retrieve",
        {"query": "canary release rollback", "tenant_context": acme_tenant.model_dump(), "top_k": 3},
        reader,
    )
    for r in result.results:
        print(f"  score={r.rerank_score:.3f}  {r.text[:70]}...")

    print("\n--- guarded_execute against the REAL 'retrieve' tool: malformed JSON ---")
    block = guarded_execute(
        "retrieve",
        '{"query": "revenue",}',  # trailing comma — a model actually does this
        reader,
        tool_use_id="demo_malformed",
    )
    print(f"  is_error={block.get('is_error', False)}: {block['content'][:160]}")

    print("\n--- guarded_execute against the REAL 'retrieve' tool: missing required field ---")
    block = guarded_execute(
        "retrieve",
        '{"top_k": 3}',  # missing "query" and "tenant_context"
        reader,
        tool_use_id="demo_schema",
    )
    print(f"  is_error={block.get('is_error', False)}: {block['content'][:200]}")

    print("\n--- guarded_execute against the REAL 'retrieve' tool: valid call, with retry/timeout budget ---")
    block = guarded_execute(
        "retrieve",
        json.dumps({"query": "canary release rollback", "tenant_context": acme_tenant.model_dump(), "top_k": 3}),
        reader,
        tool_use_id="demo_valid",
        max_attempts=3,
        timeout_s=15.0,
    )
    print(f"  is_error={block.get('is_error', False)}: {block['content'][:160]}")

    print("\n--- monitoring summary (calls_total / denials_total / p95 latency per tool) ---")
    print(json.dumps(metrics.summary(), indent=2))


if __name__ == "__main__":
    main()
