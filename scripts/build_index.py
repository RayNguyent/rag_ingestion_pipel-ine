"""
End-to-end index build: load -> clean -> chunk -> tag tenant/ACL metadata
-> embed -> write both the dense vector store and the sparse BM25 index.

In a real deployment, the (path, tenant_id, acl_roles, extra_metadata)
manifest below would come from your document management system / DB, not
be hardcoded here — this is a stand-in to demonstrate multi-tenant
ingestion end to end.
"""

from pathlib import Path

from app.chunker import split_text
from app.config import settings
from app.embeddings.tfidf_embedder import TfidfLsaEmbedder
from app.indexer import create_chunk_records, save_records
from app.loader import load_txt
from app.sparse.bm25_index import BM25Index
from app.vectorstore.numpy_store import NumpyVectorStore

# path, tenant_id, acl_roles, extra_metadata
INGEST_MANIFEST = [
    (
        "data/sample.txt",
        "acme",
        ["*"],
        {"doc_type": "engineering_notes"},
    ),
    (
        "data/tenant_b_handbook.txt",
        "globex",
        ["employee", "manager"],
        {"doc_type": "hr_handbook"},
    ),
]


def build_index() -> None:
    all_records = []

    for path, tenant_id, acl_roles, extra_metadata in INGEST_MANIFEST:
        doc = load_txt(path)
        chunks = split_text(
            doc.text,
            splitter_name="recursive",
            chunk_size=300,
            overlap=50,
        )
        records = create_chunk_records(
            document_id=doc.document_id,
            source=doc.source,
            chunks=chunks,
            tenant_id=tenant_id,
            acl_roles=acl_roles,
            extra_metadata=extra_metadata,
        )
        all_records.extend(records)
        print(f"[build_index] {path}: {len(records)} chunks (tenant={tenant_id})")

    texts = [r.text for r in all_records]
    ids = [r.chunk_id for r in all_records]
    metadatas = [r.metadata for r in all_records]
    document_ids = [r.document_id for r in all_records]

    embedder = TfidfLsaEmbedder(dim=settings.embedding_dim)
    embedder.fit(texts)
    vectors = embedder.embed(texts)

    vector_store = NumpyVectorStore()
    vector_store.upsert(
        ids=ids, vectors=vectors, metadatas=metadatas, texts=texts, document_ids=document_ids
    )

    bm25_index = BM25Index()
    bm25_index.build(ids=ids, texts=texts, metadatas=metadatas)

    Path("output").mkdir(exist_ok=True)
    vector_store.save(settings.vector_store_path)
    bm25_index.save(settings.bm25_index_path)
    embedder.save("output/embedder.pkl")
    save_records(all_records, "output/chunks.json")  # kept for debugging/back-compat

    print(f"[build_index] indexed {len(all_records)} total chunks across {len(INGEST_MANIFEST)} documents")
    print(f"[build_index] vector store -> {settings.vector_store_path}")
    print(f"[build_index] bm25 index   -> {settings.bm25_index_path}")


if __name__ == "__main__":
    build_index()
