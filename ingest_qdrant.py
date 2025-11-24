import json
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
QDRANT_URL = "https://b0842b9c-1158-4fda-9767-3e745a77cf33.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.D5iASQgZQ26FJJ9ZLhUaX2RbcVuYZ-m62VA5PKDhJ9U"
COLLECTION_NAME = "cern_demo"
CHUNKS_FILE = "chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64



def load_chunks(path):
    print(f"Loading chunks from: {path}")
    with open(path, "r") as f:
        chunks = json.load(f)
    return chunks


def ensure_collection(client, vector_size):
    """Create Qdrant collection if it does not exist."""
    existing = client.get_collections().collections
    names = [c.name for c in existing]

    if COLLECTION_NAME not in names:
        print(f"Creating Qdrant collection: {COLLECTION_NAME}")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")



def batch(iterable, bs):
    """Yield batches for faster processing."""
    for i in range(0, len(iterable), bs):
        yield iterable[i:i + bs]



def main():
    # -------------------------------------------------
    # Step 1: Load chunks
    # -------------------------------------------------
    chunks = load_chunks(CHUNKS_FILE)

    if not chunks:
        print("ERROR: chunks.json is empty!")
        return

    texts = [c["text"] for c in chunks]

    # -------------------------------------------------
    # Step 2: Load embedding model
    # -------------------------------------------------
    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Test one embedding
    test_vec = model.encode("test").tolist()
    vector_size = len(test_vec)

    # -------------------------------------------------
    # Step 3: Connect to Qdrant
    # -------------------------------------------------
    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    ensure_collection(client, vector_size)

    # -------------------------------------------------
    # Step 4: Insert embeddings in batches
    # -------------------------------------------------
    print(f"Embedding {len(texts)} chunks and uploading to Qdrant...")

    total_uploaded = 0

    for chunk_batch in tqdm(batch(chunks, BATCH_SIZE)):
        batch_texts = [c["text"] for c in chunk_batch]
        embeddings = model.encode(batch_texts).tolist()

        points = []
        for emb, ck in zip(embeddings, chunk_batch):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "text": ck["text"],
                        "page": ck.get("page", None),
                        "chunk_index": ck.get("chunk_index", None),
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        total_uploaded += len(points)

    print("------------------------------------------------")
    print(f"SUCCESS: Uploaded {total_uploaded} chunks to Qdrant.")
    print("------------------------------------------------")

    # Verify
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Qdrant now contains {count} points.")



if __name__ == "__main__":
    main()
