from app.access.guard import AccessViolation, enforce
from app.models import SearchResult, TenantContext


def test_tenant_isolation_even_for_best_matching_query(two_tenant_engine):
    """A tenant-b caller searching for tenant-a-only vocabulary must get
    zero tenant-a results back, even though those chunks would score
    highest on relevance."""
    engine, _, _ = two_tenant_engine
    ctx = TenantContext(tenant_id="tenant-b", user_id="u1", roles=["*"])

    results = engine.retrieve("quarterly revenue cloud subscriptions margin", ctx, top_k=5)

    assert len(results) > 0
    assert all(r.metadata["tenant_id"] == "tenant-b" for r in results)


def test_tenant_isolation_symmetric(two_tenant_engine):
    engine, _, _ = two_tenant_engine
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])

    results = engine.retrieve("security awareness training vacation policy", ctx, top_k=5)

    assert len(results) > 0
    assert all(r.metadata["tenant_id"] == "tenant-a" for r in results)


def test_role_scoped_content_hidden_from_non_hr_role(role_scoped_engine):
    """tenant-b content is tagged acl_roles=['hr']; a tenant-b caller
    without the 'hr' role must not see it, even for an in-tenant query."""
    engine, _, _ = role_scoped_engine
    non_hr_ctx = TenantContext(tenant_id="tenant-b", user_id="u1", roles=["employee"])

    results = engine.retrieve("security training vacation orientation", non_hr_ctx, top_k=5)

    assert results == []


def test_role_scoped_content_visible_to_hr_role(role_scoped_engine):
    engine, _, _ = role_scoped_engine
    hr_ctx = TenantContext(tenant_id="tenant-b", user_id="u1", roles=["hr"])

    results = engine.retrieve("security training vacation orientation", hr_ctx, top_k=5)

    assert len(results) > 0
    assert all(r.metadata["tenant_id"] == "tenant-b" for r in results)


def test_caller_cannot_widen_access_via_filters(two_tenant_engine):
    """Passing tenant_id in extra `filters` must NOT override the caller's
    authenticated tenant_id from TenantContext."""
    engine, _, _ = two_tenant_engine
    ctx = TenantContext(tenant_id="tenant-b", user_id="u1", roles=["*"])

    results = engine.retrieve(
        "quarterly revenue",
        ctx,
        filters={"tenant_id": "tenant-a"},  # attempted override
        top_k=5,
    )

    assert all(r.metadata["tenant_id"] == "tenant-b" for r in results)


def test_access_guard_blocks_mismatched_tenant_result():
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    leaked_result = SearchResult(
        chunk_id="x",
        document_id="doc-1",
        text="leaked content",
        metadata={"tenant_id": "tenant-b", "acl_roles": ["*"]},
    )

    try:
        enforce([leaked_result], ctx, strict=True)
        assert False, "expected AccessViolation to be raised"
    except AccessViolation:
        pass


def test_access_guard_drops_mismatched_result_in_non_strict_mode():
    ctx = TenantContext(tenant_id="tenant-a", user_id="u1", roles=["*"])
    leaked_result = SearchResult(
        chunk_id="x",
        document_id="doc-1",
        text="leaked content",
        metadata={"tenant_id": "tenant-b", "acl_roles": ["*"]},
    )

    safe = enforce([leaked_result], ctx, strict=False)
    assert safe == []
