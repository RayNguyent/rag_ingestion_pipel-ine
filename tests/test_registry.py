import pytest
from pydantic import BaseModel, ValidationError

from app.generation.bot_llm import StubLLMClient
from app.generation.pipeline import RAGAnswerEngine
from app.models import AnswerResult, SearchResult, TenantContext
from app.registry.access import (
    CallerContext,
    PermissionDeniedError,
    ingestion_caller,
    read_only_caller,
)
from app.registry.execution import execute, monitored_execute
from app.registry.metrics import Metrics
from app.registry.models import ToolKind, ToolNotFoundError, ToolSpec
from app.registry.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Local fixtures: a registry wired around the real two_tenant_engine fixture
# (from conftest.py) plus a fake write tool, so tests never touch disk or
# network (no sentence-transformers download, no persisted index files).
# ---------------------------------------------------------------------------
class RetrieveInput(BaseModel):
    query: str
    tenant_context: TenantContext
    filters: dict | None = None
    top_k: int | None = None


class RetrieveOutput(BaseModel):
    results: list[SearchResult]


class AnswerInput(BaseModel):
    query: str
    tenant_context: TenantContext
    top_k: int | None = None


class FakeIngestInput(BaseModel):
    path: str
    tenant_id: str


class FakeIngestOutput(BaseModel):
    chunks_added: int


@pytest.fixture
def wired_registry(two_tenant_engine):
    engine, _, _ = two_tenant_engine
    answer_engine = RAGAnswerEngine(engine, StubLLMClient())
    reg = ToolRegistry()

    reg.register(
        ToolSpec(
            name="retrieve",
            kind=ToolKind.READ,
            input_model=RetrieveInput,
            output_model=RetrieveOutput,
            required_scope="retrieval:read",
            fn=lambda i: RetrieveOutput(
                results=engine.retrieve(i.query, i.tenant_context, filters=i.filters, top_k=i.top_k)
            ),
        )
    )
    reg.register(
        ToolSpec(
            name="answer",
            kind=ToolKind.READ,
            input_model=AnswerInput,
            output_model=AnswerResult,
            required_scope="generation:read",
            fn=lambda i: answer_engine.answer(i.query, i.tenant_context, top_k=i.top_k),
        )
    )
    reg.register(
        ToolSpec(
            name="ingest_document",
            kind=ToolKind.WRITE,
            input_model=FakeIngestInput,
            output_model=FakeIngestOutput,
            required_scope="index:write",
            # never actually touches disk — just proves the gating works
            fn=lambda i: FakeIngestOutput(chunks_added=3),
        )
    )
    return reg


# ---------------------------------------------------------------------------
# Registry bookkeeping
# ---------------------------------------------------------------------------
def test_read_write_segregation(wired_registry):
    read_names = {t.name for t in wired_registry.read_tools()}
    write_names = {t.name for t in wired_registry.write_tools()}

    assert read_names == {"retrieve", "answer"}
    assert write_names == {"ingest_document"}
    assert read_names.isdisjoint(write_names)


def test_duplicate_registration_rejected(wired_registry):
    with pytest.raises(ValueError):
        wired_registry.register(
            ToolSpec(
                name="retrieve",
                kind=ToolKind.READ,
                input_model=RetrieveInput,
                output_model=RetrieveOutput,
                required_scope="retrieval:read",
                fn=lambda i: None,
            )
        )


def test_unknown_tool_raises_tool_not_found(wired_registry):
    with pytest.raises(ToolNotFoundError):
        wired_registry.get("delete_everything")


def test_anthropic_tool_schemas_excludes_write_tools_for_read_only_kind(wired_registry):
    schemas = wired_registry.anthropic_tool_schemas(kinds={ToolKind.READ})
    names = {s["name"] for s in schemas}

    assert names == {"retrieve", "answer"}
    assert "ingest_document" not in names


# ---------------------------------------------------------------------------
# Permission enforcement
# ---------------------------------------------------------------------------
def test_read_only_caller_can_call_read_tool(wired_registry, two_tenant_engine):
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    caller = read_only_caller(ctx)

    result = execute(
        "retrieve",
        {"query": "quarterly revenue cloud", "tenant_context": ctx.model_dump(), "top_k": 3},
        caller,
        registry=wired_registry,
    )

    assert isinstance(result, RetrieveOutput)
    assert len(result.results) > 0
    assert all(r.metadata["tenant_id"] == "tenant-a" for r in result.results)


def test_read_only_caller_denied_write_tool(wired_registry):
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    caller = read_only_caller(ctx)

    with pytest.raises(PermissionDeniedError):
        execute(
            "ingest_document",
            {"path": "data/whatever.txt", "tenant_id": "tenant-a"},
            caller,
            registry=wired_registry,
        )


def test_ingestion_caller_can_call_write_tool(wired_registry):
    ctx = TenantContext(tenant_id="tenant-a", user_id="admin-1", roles=["*"])
    caller = ingestion_caller(ctx)

    result = execute(
        "ingest_document",
        {"path": "data/whatever.txt", "tenant_id": "tenant-a"},
        caller,
        registry=wired_registry,
    )

    assert result.chunks_added == 3


def test_caller_with_no_scopes_denied_everything(wired_registry):
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    caller = CallerContext(tenant_context=ctx, scopes=frozenset())

    with pytest.raises(PermissionDeniedError):
        execute(
            "retrieve",
            {"query": "anything", "tenant_context": ctx.model_dump()},
            caller,
            registry=wired_registry,
        )


# ---------------------------------------------------------------------------
# Input validation (Pydantic contract enforcement)
# ---------------------------------------------------------------------------
def test_malformed_input_rejected_before_tool_runs(wired_registry):
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    caller = read_only_caller(ctx)

    with pytest.raises(ValidationError):
        # missing required "query" field
        execute(
            "retrieve",
            {"tenant_context": ctx.model_dump()},
            caller,
            registry=wired_registry,
        )


# ---------------------------------------------------------------------------
# Regression: wrapping retrieval in the registry must not weaken tenant
# isolation that ContextRetrievalEngine already enforces.
# ---------------------------------------------------------------------------
def test_registry_preserves_tenant_isolation(wired_registry):
    ctx = TenantContext(tenant_id="tenant-b", user_id="u1", roles=["*"])
    caller = read_only_caller(ctx)

    result = execute(
        "retrieve",
        {
            "query": "quarterly revenue cloud subscriptions",  # tenant-a vocabulary
            "tenant_context": ctx.model_dump(),
            "filters": {"tenant_id": "tenant-a"},  # attempted override
            "top_k": 5,
        },
        caller,
        registry=wired_registry,
    )

    assert all(r.metadata["tenant_id"] == "tenant-b" for r in result.results)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
def test_monitored_execute_records_success_and_denials(wired_registry):
    metrics = Metrics()
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    reader = read_only_caller(ctx)

    monitored_execute(
        "retrieve",
        {"query": "quarterly revenue", "tenant_context": ctx.model_dump(), "top_k": 3},
        reader,
        registry=wired_registry,
        metrics=metrics,
    )

    with pytest.raises(PermissionDeniedError):
        monitored_execute(
            "ingest_document",
            {"path": "x.txt", "tenant_id": "tenant-a"},
            reader,
            registry=wired_registry,
            metrics=metrics,
        )

    summary = metrics.summary()
    assert summary["retrieve"]["calls_total"] == 1
    assert summary["retrieve"]["denials_total"] == 0
    assert summary["ingest_document"]["calls_total"] == 1
    assert summary["ingest_document"]["denials_total"] == 1
    assert summary["retrieve"]["p95_latency_ms"] >= 0
