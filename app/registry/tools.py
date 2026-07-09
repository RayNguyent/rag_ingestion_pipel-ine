"""
Wires the project's actual retrieval/generation/ingestion entrypoints into
the tool registry. Import `register_all()` once at process startup (see
scripts/registry_demo.py) — nothing else should call ContextRetrievalEngine,
RAGAnswerEngine, or ingest_single_document directly once this is done.
"""

from pydantic import BaseModel, Field

from app.generation.loader import load_answer_engine
from app.incremental_ingest import ingest_single_document
from app.models import AnswerResult, SearchResult, TenantContext
from app.registry.models import ToolKind, ToolSpec
from app.registry.registry import ToolRegistry
from app.registry.registry import registry as default_registry
from app.retrieval.loader import load_engine


# ---------------------------------------------------------------------------
# retrieve (READ)
# ---------------------------------------------------------------------------
class RetrieveInput(BaseModel):
    query: str = Field(..., description="Search query text")
    tenant_context: TenantContext
    filters: dict | None = Field(None, description="Optional extra metadata filters")
    top_k: int | None = Field(None, ge=1, le=50)


class RetrieveOutput(BaseModel):
    results: list[SearchResult]


# ---------------------------------------------------------------------------
# answer (READ — it calls the LLM but never mutates stored state)
# ---------------------------------------------------------------------------
class AnswerInput(BaseModel):
    query: str
    tenant_context: TenantContext
    filters: dict | None = None
    top_k: int | None = Field(None, ge=1, le=50)


# AnswerResult is already a Pydantic model defined in app/models.py — reused
# directly as the output_model.


# ---------------------------------------------------------------------------
# ingest_document (WRITE — mutates the persisted vector store + BM25 index)
# ---------------------------------------------------------------------------
class IngestDocumentInput(BaseModel):
    path: str = Field(..., description="Path to the source text file to ingest")
    tenant_id: str
    acl_roles: list[str] | None = None
    extra_metadata: dict | None = None


class IngestDocumentOutput(BaseModel):
    document_id: str
    source: str
    tenant_id: str
    chunks_added: int
    total_chunks_in_index: int


def register_all(registry: ToolRegistry = default_registry) -> None:
    retrieval_engine = load_engine()
    answer_engine = load_answer_engine()

    registry.register(
        ToolSpec(
            name="retrieve",
            kind=ToolKind.READ,
            input_model=RetrieveInput,
            output_model=RetrieveOutput,
            required_scope="retrieval:read",
            description="Hybrid dense+sparse retrieval, tenant/ACL-scoped, reranked.",
            fn=lambda i: RetrieveOutput(
                results=retrieval_engine.retrieve(
                    i.query, i.tenant_context, filters=i.filters, top_k=i.top_k
                )
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="answer",
            kind=ToolKind.READ,
            input_model=AnswerInput,
            output_model=AnswerResult,
            required_scope="generation:read",
            description="Retrieve + generate a cited answer via the configured LLM.",
            fn=lambda i: answer_engine.answer(
                i.query, i.tenant_context, filters=i.filters, top_k=i.top_k
            ),
        )
    )

    registry.register(
        ToolSpec(
            name="ingest_document",
            kind=ToolKind.WRITE,
            input_model=IngestDocumentInput,
            output_model=IngestDocumentOutput,
            required_scope="index:write",
            description="Chunk, embed, and upsert a new document into the persisted index.",
            fn=lambda i: ingest_single_document(
                path=i.path,
                tenant_id=i.tenant_id,
                acl_roles=i.acl_roles,
                extra_metadata=i.extra_metadata,
            ),
        )
    )
