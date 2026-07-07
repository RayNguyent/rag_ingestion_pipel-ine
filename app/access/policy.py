from typing import Any

from app.models import TenantContext


def build_filters(
    tenant_context: TenantContext,
    extra_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Always injects tenant_id and acl_roles from the authenticated caller —
    these are never taken from caller-supplied extra_filters, so a request
    can't widen its own access by passing e.g. {"tenant_id": "other-tenant"}.
    """
    filters: dict[str, Any] = dict(extra_filters or {})
    filters["tenant_id"] = tenant_context.tenant_id
    filters["acl_roles"] = tenant_context.roles
    return filters
