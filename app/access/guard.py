import logging

from app.filters import matches_filters
from app.models import SearchResult, TenantContext

logger = logging.getLogger(__name__)


class AccessViolation(Exception):
    """Raised when a result that should have been filtered out slips through.
    This should never fire in normal operation — its purpose is to catch
    bugs in the store/index filtering, not to be a routine code path."""


def enforce(
    results: list[SearchResult],
    tenant_context: TenantContext,
    strict: bool = True,
) -> list[SearchResult]:
    """
    Second, independent check that every result actually belongs to the
    caller's tenant and is visible under their roles. The vector store and
    BM25 index already filter before returning candidates; this exists so
    that a filtering bug in one backend can't silently leak data — it fails
    loudly instead.
    """
    safe_results = []
    for result in results:
        allowed = matches_filters(
            result.metadata,
            {"tenant_id": tenant_context.tenant_id, "acl_roles": tenant_context.roles},
        )
        if not allowed:
            message = (
                f"Access guard blocked chunk_id={result.chunk_id} "
                f"(tenant={result.metadata.get('tenant_id')}) "
                f"for caller tenant={tenant_context.tenant_id}, user={tenant_context.user_id}"
            )
            logger.error(message)
            if strict:
                raise AccessViolation(message)
            continue
        safe_results.append(result)
    return safe_results
