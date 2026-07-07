from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document")
    source: str
    text: str


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str
    text: str
    metadata: dict
    # metadata is expected to always carry, at minimum:
    #   - tenant_id: str        -> owning tenant, enforced at every retrieval layer
    #   - acl_roles: list[str]  -> roles allowed to see this chunk (e.g. ["*"] for public)
    # Anything else (source, chunk_number, doc_type, ...) is free-form and filterable.


class TenantContext(BaseModel):
    """Request-scoped identity used to authorize a retrieval call."""

    tenant_id: str = Field(..., description="Tenant the caller belongs to")
    user_id: str = Field(..., description="Caller's user id, for auditing")
    roles: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Roles held by the caller; '*' matches any acl_roles entry",
    )


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: dict
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None