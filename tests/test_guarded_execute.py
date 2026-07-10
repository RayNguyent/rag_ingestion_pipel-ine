import pytest
from pydantic import BaseModel

from app.models import TenantContext
from app.registry.access import ingestion_caller, read_only_caller
from app.registry.errors import TransientConnectionError
from app.registry.execution import guarded_execute
from app.registry.metrics import Metrics
from app.registry.models import ToolKind, ToolSpec
from app.registry.registry import ToolRegistry


class EchoInput(BaseModel):
    query: str
    top_k: int = 5


class EchoOutput(BaseModel):
    query: str
    top_k: int


class BadOutput(BaseModel):
    # deliberately incompatible with what the flaky/echo tool returns,
    # to exercise the OutputContractError path
    unexpected_required_field: str


@pytest.fixture
def registry_with_flaky_tool():
    reg = ToolRegistry()
    reg.register(
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

    def flaky_fn(i: EchoInput):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientConnectionError("simulated network blip")
        return EchoOutput(query=i.query, top_k=i.top_k)

    reg.register(
        ToolSpec(
            name="flaky",
            kind=ToolKind.READ,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="retrieval:read",
            fn=flaky_fn,
        )
    )
    reg.register(
        ToolSpec(
            name="always_hangs",
            kind=ToolKind.READ,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="retrieval:read",
            fn=lambda i: __import__("time").sleep(5) or EchoOutput(query=i.query, top_k=i.top_k),
        )
    )
    reg.register(
        ToolSpec(
            name="write_tool",
            kind=ToolKind.WRITE,
            input_model=EchoInput,
            output_model=EchoOutput,
            required_scope="index:write",
            fn=lambda i: EchoOutput(query=i.query, top_k=i.top_k),
        )
    )
    reg.register(
        ToolSpec(
            name="broken_contract",
            kind=ToolKind.READ,
            input_model=EchoInput,
            output_model=BadOutput,
            required_scope="retrieval:read",
            # returns something that will never satisfy BadOutput's schema
            fn=lambda i: EchoOutput(query=i.query, top_k=i.top_k),
        )
    )
    return reg, attempts


def _reader():
    return read_only_caller(TenantContext(tenant_id="acme", user_id="u1", roles=["*"]))


def _writer():
    return ingestion_caller(TenantContext(tenant_id="acme", user_id="admin", roles=["*"]))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_successful_call_returns_non_error_tool_result_block(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute(
        "echo", '{"query": "hi", "top_k": 2}', _reader(), tool_use_id="call_1", registry=reg
    )
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert "is_error" not in block
    assert '"query":"hi"' in block["content"] or '"query": "hi"' in block["content"]


# ---------------------------------------------------------------------------
# Phase 1: malformed JSON / schema validation
# ---------------------------------------------------------------------------
def test_malformed_json_returns_error_block_without_raising(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute("echo", '{"query": "hi",}', _reader(), registry=reg)
    assert block["is_error"] is True
    assert "not valid JSON" in block["content"]


def test_schema_violation_names_the_field(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute("echo", '{"top_k": 2}', _reader(), registry=reg)
    assert block["is_error"] is True
    assert "query" in block["content"]


def test_unknown_tool_returns_error_block_listing_available_tools(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute("does_not_exist", "{}", _reader(), registry=reg)
    assert block["is_error"] is True
    assert "does_not_exist" in block["content"]
    assert "echo" in block["content"]


# ---------------------------------------------------------------------------
# Permission enforcement still applies
# ---------------------------------------------------------------------------
def test_read_only_caller_denied_write_tool_via_guarded_execute(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute(
        "write_tool", '{"query": "hi"}', _reader(), registry=reg
    )
    assert block["is_error"] is True


def test_permission_denial_recorded_as_denied_not_error_in_metrics(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    metrics = Metrics()
    guarded_execute("write_tool", '{"query": "hi"}', _reader(), registry=reg, metrics=metrics)
    summary = metrics.summary()["write_tool"]
    assert summary["denials_total"] == 1
    assert summary["errors_total"] == 0


# ---------------------------------------------------------------------------
# Phase 3: resilience — retries succeed transparently, timeouts surface cleanly
# ---------------------------------------------------------------------------
def test_flaky_tool_succeeds_after_retries(registry_with_flaky_tool):
    reg, attempts = registry_with_flaky_tool
    block = guarded_execute(
        "flaky", '{"query": "hi", "top_k": 1}', _reader(), registry=reg, max_attempts=5
    )
    assert "is_error" not in block
    assert attempts["n"] == 3


def test_flaky_tool_retries_recorded_in_metrics(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    metrics = Metrics()
    guarded_execute(
        "flaky", '{"query": "hi", "top_k": 1}', _reader(), registry=reg, metrics=metrics, max_attempts=5
    )
    summary = metrics.summary()["flaky"]
    assert summary["retries_total"] == 2


def test_hanging_tool_times_out_cleanly(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute(
        "always_hangs", '{"query": "hi"}', _reader(), registry=reg, max_attempts=1, timeout_s=0.05
    )
    assert block["is_error"] is True
    assert "timed out" in block["content"]


def test_timeout_recorded_in_metrics(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    metrics = Metrics()
    guarded_execute(
        "always_hangs", '{"query": "hi"}', _reader(), registry=reg, metrics=metrics,
        max_attempts=1, timeout_s=0.05,
    )
    summary = metrics.summary()["always_hangs"]
    assert summary["timeouts_total"] == 1


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_broken_output_contract_surfaces_as_clean_error_not_stack_trace(registry_with_flaky_tool):
    reg, _ = registry_with_flaky_tool
    block = guarded_execute("broken_contract", '{"query": "hi"}', _reader(), registry=reg)
    assert block["is_error"] is True
    assert "bug in the tool" in block["content"]
