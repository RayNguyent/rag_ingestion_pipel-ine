import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.embeddings.base import Embedder


class TfidfLsaEmbedder(Embedder):
    """
    Dense embedder that needs no network access or model weights: TF-IDF
    vectors are projected down to `dim` dimensions with LSA (TruncatedSVD).
    This is a reasonable offline default and a real dense representation
    (not a placeholder), but it captures lexical/co-occurrence structure
    rather than deep semantics. For production-grade semantic recall, swap
    in a sentence-transformers or hosted embedding API behind this same
    Embedder interface — nothing else in the pipeline needs to change.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectorizer = TfidfVectorizer()
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        tfidf = self.vectorizer.fit_transform(texts)
        n_components = min(self.dim, max(2, min(tfidf.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf)
        self.dim = n_components
        self._fitted = True

    def _transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError("Embedder must be fit() before embedding text.")
        tfidf = self.vectorizer.transform(texts)
        vectors = self.svd.transform(tfidf)
        return normalize(vectors, norm="l2", axis=1)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim},
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self.svd = state["svd"]
        self.dim = state["dim"]
        self._fitted = True
