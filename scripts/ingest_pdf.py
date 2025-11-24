import base64
import json
import os
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
DOCLING_URL = "http://docling-serve:5001/v1/chunk/hierarchical/file"
PDF_PATH = "/app/data/CERN_Yellow_Report_357576.pdf"   # IMPORTANT: /app == container
COLLECTION_NAME = "cern_demo"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 1. LOAD PDF & SEND TO DOCLING
# -----------------------------
print("📌 Loading PDF...")
with open(PDF_PATH, "rb") as f:
    pdf_bytes = f.read()

files = {"files": ("document.pdf", pdf_bytes, "application/pdf")}

print("📌 Sending PDF to Docling chunking API...")
response = requests.post(
    DOCLING_URL,
    files=files,
    data={"include_converted_doc": "false", "target_type": "inbody"}
)

if response.status_code != 200:
    print("❌ Docling Error:", response.text)
    exit()

doc_output = response.json()

chunks = doc_output.get("chunks", [])
print(f"📌 Received {len(chunks)} chunks from Docling")

# -----------------------------
# 2. CONNECT TO QDRANT CLOUD
# -----------------------------
print("📌 Connecting to Qdrant Cloud...")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# -----------------------------
# 3. EMBED & UPLOAD CHUNKS
# -----------------------------
points = []
print("📌 Embedding and uploading chunks to Qdrant...")

for i, chunk in enumerate(chunks):
    text = chunk.get("text", "").strip()

    if not text:
        continue

    embedding = model.encode(text).tolist()

    point = PointStruct(
        id=i,
        vector=embedding,
        payload={
            "text": text,
            "page_numbers": chunk.get("page_numbers", []),
            "headings": chunk.get("headings", []),
            "chunk_index": chunk.get("chunk_index", i),
            "filename": chunk.get("filename", "N/A"),
        }
    )

    points.append(point)

# Upload in batches
BATCH = 100
for i in range(0, len(points), BATCH):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points[i:i+BATCH]
    )
    print(f"📌 Uploaded batch {i//BATCH + 1}")

print("✅ Ingestion completed successfully.")
