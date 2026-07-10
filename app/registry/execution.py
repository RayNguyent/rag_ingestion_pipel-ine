import time

from pydantic import BaseModel, ValidationError

from app.registry.access import CallerContext, PermissionDeniedError, require_scope
from app.registry.errors import (
    OutputContractError,
    TimeoutToolError,
    ToolCallError,
    ToolNotFoundRecoveryError,
)
from app.registry.metrics import Metrics, metrics as default_metrics
from app.registry.models import ToolNotFoundError
from app.registry.recovery import build_tool_result_block
from app.registry.registry import ToolRegistry, registry as default_registry
from app.registry.resilience import call_with_resilience
from app.registry.validation import parse_and_validate


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
    # Resolve kind upfront where possible so a denial/not-found still logs
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


def guarded_execute(
    tool_name: str,
    raw_args: str | dict,
    ctx: CallerContext,
    *,
    tool_use_id: str | None = None,
    registry: ToolRegistry = default_registry,
    metrics: Metrics = default_metrics,
    max_attempts: int = 3,
    timeout_s: float = 10.0,
) -> dict:
    """
    The entrypoint an agent loop should actually call when `raw_args`
    comes straight from a model's tool_use block (a raw string, not a
    trusted dict) and the tool itself might be slow, flaky, or rate
    limited.

    Unlike execute()/monitored_execute(), this NEVER raises for expected
    failure modes (bad JSON, schema violations, permission denial,
    timeouts, rate limits) — it always returns an Anthropic-shaped
    tool_result block, success or error, ready to append to the next
    message sent back to the model. It only raises for truly unexpected
    bugs.

    Order of operations, each step guarding the next:
    1. look up the tool                      -> ToolNotFoundRecoveryError
    2. check the caller's scope               -> PermissionDeniedError
    3. parse + validate raw_args              -> MalformedJSONError / SchemaValidationError
    4. run the tool, with timeout + retries   -> TimeoutToolError / RateLimitedError / TransientConnectionError
    5. validate the tool's return value        -> OutputContractError
    """
    result_id = tool_use_id or tool_name
    start = time.perf_counter()
    ok = False
    denied = False
    retries = 0
    timed_out = False
    kind = "unknown"
    required_scope = "unknown"

    try:
        spec = registry.get(tool_name)
    except ToolNotFoundError:
        error = ToolNotFoundRecoveryError(
            f"No tool named '{tool_name}' is registered. "
            f"Available tools: {[t.name for t in registry.all_tools()]}"
        )
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record(
            tool_name=tool_name, kind=kind, caller_id=ctx.tenant_context.user_id,
            tenant_id=ctx.tenant_context.tenant_id, scope=required_scope,
            ok=False, denied=False, duration_ms=duration_ms,
        )
        return build_tool_result_block(result_id, error=error)

    kind = spec.kind.value
    required_scope = spec.required_scope
    error: Exception | None = None

    try:
        require_scope(ctx, required_scope, tool_name)
        validated_input = parse_and_validate(raw_args, spec.input_model)

        attempts_used = 0

        def run_once(_validated_input=validated_input):
            nonlocal attempts_used
            attempts_used += 1
            return spec.fn(_validated_input)

        raw_result = call_with_resilience(run_once, max_attempts=max_attempts, timeout_s=timeout_s)
        retries = attempts_used - 1

        try:
            validated_output = spec.output_model.model_validate(raw_result)
        except ValidationError as e:
            raise OutputContractError(
                f"Tool '{tool_name}' ran successfully but its result didn't match "
                f"its own output schema ({e}). This is a bug in the tool, not "
                f"something you can fix by changing your arguments."
            ) from e

        ok = True
        return build_tool_result_block(result_id, error=None, result=validated_output)

    except PermissionDeniedError as e:
        denied = True
        error = e
        return build_tool_result_block(result_id, error=error)
    except TimeoutToolError as e:
        timed_out = True
        error = e
        return build_tool_result_block(result_id, error=error)
    except ToolCallError as e:
        error = e
        return build_tool_result_block(result_id, error=error)
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
            retries=retries,
            timed_out=timed_out,
        )
