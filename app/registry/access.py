from pydantic import BaseModel, Field

from app.models import TenantContext


class PermissionDeniedError(Exception):
    """Raised when a caller lacks the scope required to invoke a tool."""


class CallerContext(BaseModel):
    """
    Identity used to authorize a *tool call*, distinct from TenantContext
    (which authorizes which *chunks* a retrieval may return). A single
    caller carries both: TenantContext still drives row-level ACL
    filtering inside ContextRetrievalEngine; `scopes` here drives which
    tools in the registry they're even allowed to reach.

    Example: a chat-widget backend calling on behalf of an end user gets
    scopes={"retrieval:read", "generation:read"} — it can never reach
    "index:write", no matter what TenantContext it's holding.
    """

    tenant_context: TenantContext
    scopes: frozenset[str] = Field(default_factory=frozenset)

    model_config = {"arbitrary_types_allowed": True}


# Convenience presets so callers don't have to hand-build scope sets.
READ_ONLY_SCOPES = frozenset({"retrieval:read", "generation:read", "eval:read"})
INGESTION_SCOPES = READ_ONLY_SCOPES | frozenset({"index:write"})


def read_only_caller(tenant_context: TenantContext) -> CallerContext:
    return CallerContext(tenant_context=tenant_context, scopes=READ_ONLY_SCOPES)


def ingestion_caller(tenant_context: TenantContext) -> CallerContext:
    return CallerContext(tenant_context=tenant_context, scopes=INGESTION_SCOPES)


def require_scope(ctx: CallerContext, required_scope: str, tool_name: str) -> None:
    if required_scope not in ctx.scopes:
        raise PermissionDeniedError(
            f"Caller user_id={ctx.tenant_context.user_id!r} "
            f"tenant={ctx.tenant_context.tenant_id!r} lacks scope "
            f"{required_scope!r} required for tool {tool_name!r} "
            f"(has: {sorted(ctx.scopes)})"
        )
