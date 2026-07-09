"""
Incremental ingestion of a single document into an already-persisted
index. scripts/build_index.py rebuilds everything from the manifest from
scratch; this module supports adding one more document afterwards without
re-embedding/re-chunking everything that came before.

This is the "write" side of the pipeline — it mutates the vector store
and rebuilds the BM25 corpus on disk — which is exactly why it's wired up
as a WRITE tool in app/registry/tools.py rather than something an agent
or a read-only caller can invoke directly.
"""

from app.chunker import split_text
from app.config import settings
from app.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.indexer import create_chunk_records
from app.loader import load_txt
from app.sparse.bm25_index import BM25Index
from app.vectorstore.numpy_store import NumpyVectorStore


def ingest_single_document(
    path: str,
    tenant_id: str,
    acl_roles: list[str] | None = None,
    extra_metadata: dict | None = None,
    splitter_name: str = "recursive",
    chunk_size: int = 300,
    overlap: int = 50,
) -> dict:
    """
    Loads the persisted vector store + BM25 index + embedder, chunks and
    embeds `path` as a new document scoped to `tenant_id`, upserts it into
    the vector store, rebuilds the BM25 corpus (BM25Okapi has no
    incremental-add API, so this step is a full rebuild over the combined
    corpus), and saves both back to disk.

    Returns a dict summary rather than raising on partial failure so the
    calling tool can report a clean status either way.
    """
    embedder = SentenceTransformerEmbedder()
    embedder.load("output/embedder.pkl")

    vector_store = NumpyVectorStore()
    vector_store.load(settings.vector_store_path)

    doc = load_txt(path)
    chunks = split_text(doc.text, splitter_name=splitter_name, chunk_size=chunk_size, overlap=overlap)
    records = create_chunk_records(
        document_id=doc.document_id,
        source=doc.source,
        chunks=chunks,
        tenant_id=tenant_id,
        acl_roles=acl_roles,
        extra_metadata=extra_metadata,
    )

    texts = [r.text for r in records]
    ids = [r.chunk_id for r in records]
    metadatas = [r.metadata for r in records]
    document_ids = [r.document_id for r in records]

    vectors = embedder.embed(texts)
    vector_store.upsert(ids=ids, vectors=vectors, metadatas=metadatas, texts=texts, document_ids=document_ids)

    # BM25 must be rebuilt over the FULL corpus (old + new) — vector_store
    # now holds that combined set after the upsert above, so it's the
    # source of truth here rather than reloading the old BM25 pickle.
    bm25_index = BM25Index()
    bm25_index.build(ids=vector_store.ids, texts=vector_store.texts, metadatas=vector_store.metadatas)

    vector_store.save(settings.vector_store_path)
    bm25_index.save(settings.bm25_index_path)

    return {
        "document_id": doc.document_id,
        "source": doc.source,
        "tenant_id": tenant_id,
        "chunks_added": len(records),
        "total_chunks_in_index": len(vector_store.ids),
    }
