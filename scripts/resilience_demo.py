"""
Demonstrates guarded_execute() against real registered tools:

  1. malformed JSON args        -> clean recovery message, no crash
  2. schema-invalid args        -> recovery message names the bad field
  3. unknown tool name          -> recovery message lists what IS available
  4. read-only caller, write tool -> permission denial as a tool_result
  5. a flaky tool that fails twice then succeeds -> transparent retry
  6. a tool that always hangs   -> clean timeout message, bounded wait
  7. final metrics summary, including retries_total / timeouts_total

Run with: python -m scripts.resilience_demo
"""

import json
import time

from app.models import TenantContext
from app.registry.access import ingestion_caller, read_only_caller
from app.registry.errors import TransientConnectionError
from app.registry.execution import guarded_execute
from app.registry.metrics import metrics
from app.registry.models import ToolKind, ToolSpec
from app.registry.registry import registry
from pydantic import BaseModel


class EchoInput(BaseModel):
    query: str
    top_k: int = 5


class EchoOutput(BaseModel):
    query: str
    top_k: int


def register_demo_tools() -> None:
    registry.register(
        ToolSpec(
            name="echo",
            kind=ToolKind.READ,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="retrieval:read",
            fn=lambda i: EchoOutput(query=i.query, top_k=i.top_k),
        )
    )

    attempts = {"n": 0}

    def flaky_fn(i: EchoInput) -> EchoOutput:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientConnectionError(f"simulated blip on attempt {attempts['n']}")
        return EchoOutput(query=i.query, top_k=i.top_k)

    registry.register(
        ToolSpec(
            name="flaky_search",
            kind=ToolKind.READ,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="retrieval:read",
            fn=flaky_fn,
        )
    )

    registry.register(
        ToolSpec(
            name="hanging_search",
            kind=ToolKind.READ,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="retrieval:read",
            fn=lambda i: time.sleep(5),
        )
    )

    registry.register(
        ToolSpec(
            name="ingest_document",
            kind=ToolKind.WRITE,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="index:write",
            fn=lambda i: EchoOutput(query=i.query, top_k=i.top_k),
        )
    )


def show(label: str, block: dict) -> None:
    status = "ERROR" if block.get("is_error") else "OK"
    print(f"\n--- {label} [{status}] ---")
    print(f"  {block['content'][:200]}")


def main() -> None:
    register_demo_tools()
    reader = read_only_caller(TenantContext(tenant_id="acme", user_id="analyst-1", roles=["*"]))
    writer = ingestion_caller(TenantContext(tenant_id="acme", user_id="admin-1", roles=["*"]))

    show(
        "1. malformed JSON",
        guarded_execute("echo", '{"query": "hello",}', reader, tool_use_id="c1"),
    )

    show(
        "2. schema violation (missing required field)",
        guarded_execute("echo", '{"top_k": 3}', reader, tool_use_id="c2"),
    )

    show(
        "3. unknown tool name",
        guarded_execute("delete_everything", "{}", reader, tool_use_id="c3"),
    )

    show(
        "4. read-only caller calling a write tool",
        guarded_execute("ingest_document", '{"query": "new doc"}', reader, tool_use_id="c4"),
    )

    show(
        "5. same write tool, correctly scoped caller",
        guarded_execute("ingest_document", '{"query": "new doc"}', writer, tool_use_id="c5"),
    )

    show(
        "6. flaky tool — fails twice, succeeds on 3rd attempt",
        guarded_execute(
            "flaky_search", '{"query": "revenue"}', reader, tool_use_id="c6", max_attempts=5
        ),
    )

    show(
        "7. tool that always hangs — clean timeout instead of an infinite wait",
        guarded_execute(
            "hanging_search", '{"query": "revenue"}', reader, tool_use_id="c7",
            max_attempts=1, timeout_s=0.5,
        ),
    )

    print("\n--- metrics summary (calls / denials / retries / timeouts / latency, per tool) ---")
    print(json.dumps(metrics.summary(), indent=2))


if __name__ == "__main__":
    main()
