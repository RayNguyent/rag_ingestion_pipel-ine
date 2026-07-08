# app/embeddings/sentence_transformer.py
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize

from app.embeddings.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def fit(self, texts: list[str]) -> None:
        # no-op (pretrained model)
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = SentenceTransformer(path)
        self.dim = self.model.get_embedding_dimension()