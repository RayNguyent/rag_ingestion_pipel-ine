import pytest

from app.embeddings.tfidf_embedder import TfidfLsaEmbedder
from app.hybrid.retriever import HybridRetriever
from app.indexer import create_chunk_records
from app.rerank.reranker import LexicalOverlapReranker
from app.retrieval.engine import ContextRetrievalEngine
from app.sparse.bm25_index import BM25Index
from app.vectorstore.numpy_store import NumpyVectorStore

TENANT_A_DOCS = [
    "Quarterly revenue grew 12 percent driven by cloud subscriptions.",
    "The finance team closes the books on the fifth business day.",
    "Cloud infrastructure costs were the largest driver of margin pressure.",
]

TENANT_B_DOCS = [
    "Employees must complete security awareness training annually.",
    "The vacation policy allows twenty days of paid time off per year.",
    "New hires receive a laptop and badge during orientation.",
]


def _build_engine(acl_roles_a=None, acl_roles_b=None):
    all_texts, all_ids, all_metadatas, all_doc_ids = [], [], [], []

    for tenant, docs, acl_roles in [
        ("tenant-a", TENANT_A_DOCS, acl_roles_a or ["*"]),
        ("tenant-b", TENANT_B_DOCS, acl_roles_b or ["*"]),
    ]:
        records = create_chunk_records(
            document_id=f"{tenant}-doc-1",
            source=f"{tenant}.txt",
            chunks=docs,
            tenant_id=tenant,
            acl_roles=acl_roles,
        )
        for r in records:
            all_texts.append(r.text)
            all_ids.append(r.chunk_id)
            all_metadatas.append(r.metadata)
            all_doc_ids.append(r.document_id)

    embedder = TfidfLsaEmbedder(dim=32)
    embedder.fit(all_texts)
    vectors = embedder.embed(all_texts)

    vector_store = NumpyVectorStore()
    vector_store.upsert(all_ids, vectors, all_metadatas, all_texts, all_doc_ids)

    bm25_index = BM25Index()
    bm25_index.build(all_ids, all_texts, all_metadatas)

    hybrid_retriever = HybridRetriever(vector_store, bm25_index, embedder)
    engine = ContextRetrievalEngine(hybrid_retriever, LexicalOverlapReranker())
    return engine, all_ids, all_metadatas


@pytest.fixture
def two_tenant_engine():
    engine, ids, metadatas = _build_engine()
    return engine, ids, metadatas


@pytest.fixture
def role_scoped_engine():
    # tenant-b content restricted to the "hr" role only
    engine, ids, metadatas = _build_engine(acl_roles_b=["hr"])
    return engine, ids, metadatas
