from app.config import settings
from app.embeddings.tfidf_embedder import TfidfLsaEmbedder
from app.hybrid.retriever import HybridRetriever
from app.rerank.reranker import LexicalOverlapReranker
from app.retrieval.engine import ContextRetrievalEngine
from app.sparse.bm25_index import BM25Index
from app.vectorstore.numpy_store import NumpyVectorStore


def load_engine() -> ContextRetrievalEngine:
    """Loads the persisted index artifacts built by scripts/build_index.py
    and wires them into a ready-to-query ContextRetrievalEngine."""

    embedder = TfidfLsaEmbedder(dim=settings.embedding_dim)
    embedder.load("output/embedder.pkl")

    vector_store = NumpyVectorStore()
    vector_store.load(settings.vector_store_path)

    bm25_index = BM25Index()
    bm25_index.load(settings.bm25_index_path)

    hybrid_retriever = HybridRetriever(vector_store, bm25_index, embedder)
    reranker = LexicalOverlapReranker()

    return ContextRetrievalEngine(hybrid_retriever, reranker)
