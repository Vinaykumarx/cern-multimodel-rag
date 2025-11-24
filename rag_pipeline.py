import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "cern_demo")

# -----------------------------
# INITIALIZE CLIENT & EMBEDDINGS
# -----------------------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# RAG QUERY FUNCTION
# -----------------------------
def rag_query(query: str, top_k=5):
    """Search Qdrant Cloud and return best matching chunks."""
    query_vector = embedder.encode(query).tolist()

    search_result = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k
    )

    sources = []
    answer_text = []

    for res in search_result:
        payload = res.payload
        text = payload.get("text", "")
        score = float(res.score)

        answer_text.append(text)
        sources.append({
            "text": text,
            "page_numbers": payload.get("page_numbers", []),
            "headings": payload.get("headings", []),
            "chunk_index": payload.get("chunk_index"),
            "score": score,
        })

    final_answer = "\n\n".join(answer_text[:3])

    return {
        "answer": final_answer,
        "sources": sources
    }
