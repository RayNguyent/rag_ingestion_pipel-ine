import time

from pydantic import BaseModel

from app.registry.access import CallerContext, PermissionDeniedError, require_scope
from app.registry.metrics import Metrics, metrics as default_metrics
from app.registry.registry import ToolRegistry, registry as default_registry


def execute(
    tool_name: str,
    raw_input: dict,
    ctx: CallerContext,
    *,
    registry: ToolRegistry = default_registry,
) -> BaseModel:
    """
    The only sanctioned way to run a registered tool:
    1. look up the tool (raises ToolNotFoundError if unknown)
    2. check the caller's scope covers the tool's required_scope
    3. validate raw_input against the tool's Pydantic input model
    4. run it
    5. validate the return value against the tool's Pydantic output model

    Nothing here is optional — skipping any step is exactly the bug this
    chokepoint exists to prevent.
    """
    spec = registry.get(tool_name)

    require_scope(ctx, spec.required_scope, tool_name)

    validated_input = spec.input_model.model_validate(raw_input)
    result = spec.fn(validated_input)
    return spec.output_model.model_validate(result)


def monitored_execute(
    tool_name: str,
    raw_input: dict,
    ctx: CallerContext,
    *,
    registry: ToolRegistry = default_registry,
    metrics: Metrics = default_metrics,
) -> BaseModel:
    """Same as execute(), plus latency/outcome tracking for every call —
    including denied and errored calls, not just successful ones."""
    start = time.perf_counter()
    ok = False
    denied = False
    # Resolve kind up front where possible so a denial/not-found still logs
    # something identifiable; fall back to "unknown" if the name is bad.
    try:
        spec = registry.get(tool_name)
        kind = spec.kind.value
        required_scope = spec.required_scope
    except Exception:
        kind = "unknown"
        required_scope = "unknown"

    try:
        result = execute(tool_name, raw_input, ctx, registry=registry)
        ok = True
        return result
    except PermissionDeniedError:
        denied = True
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record(
            tool_name=tool_name,
            kind=kind,
            caller_id=ctx.tenant_context.user_id,
            tenant_id=ctx.tenant_context.tenant_id,
            scope=required_scope,
            ok=ok,
            denied=denied,
            duration_ms=duration_ms,
        )
