from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_")

    # Embeddings
    embedding_dim: int = 128
    embedder_backend: str = "tfidf_lsa"  # swap to "sentence_transformers" / "openai" later

    # Vector store
    vector_store_path: str = "output/vector_store"

    # Sparse (BM25)
    bm25_index_path: str = "output/bm25_index.pkl"

    # Hybrid fusion
    fusion_method: str = "rrf"  # "rrf" or "weighted"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    # Retrieval
    candidate_pool_size: int = 20  # how many fused candidates go into reranking
    default_top_k: int = 5

    # Access control
    enforce_tenant_isolation: bool = True


settings = Settings()
