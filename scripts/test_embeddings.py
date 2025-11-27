# scripts/test_embeddings.py

import sys
from pathlib import Path

# Ensure project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.embedder import SentenceEmbedder  # noqa: E402


def main():
    embedder = SentenceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding dimension:", embedder.dim)

    texts = [
        "This is a test sentence.",
        "Radiation damage in silicon detectors.",
    ]
    vecs = embedder.embed(texts)
    print("Embeddings shape:", vecs.shape)
    print("First vector (first 10 dims):", vecs[0][:10])


if __name__ == "__main__":
    main()
