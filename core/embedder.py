# core/embedder.py

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"[Embedder] Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        emb = self.model.encode(texts, show_progress_bar=False)
        return np.array(emb, dtype="float32")
