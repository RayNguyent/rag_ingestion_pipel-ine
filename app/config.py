from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra = "ignore")

    # Embeddings
    embedding_dim: int = 128
    embedder_backend: str = "sentence_transformers"  # swap to "sentence_transformers" / "openai" later

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
    
    # Generation (LLM + Prompt)
    llm_backend = str = "anthropic"
    llm_model = str = "claude-sonnet-5"
    llm_max_tokens: int = 1024
    max_context_chars: int = 6000
    anthropic_api_key: str | None = Field (default=None, validation_alias="ANTHROPIC_API_KEY")


settings = Settings()
