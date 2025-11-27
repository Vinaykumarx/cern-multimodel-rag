# core/rag_pipeline.py

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List
from io import BytesIO

import numpy as np
import fitz
from PIL import Image

from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

from core.vector_store_lance import LanceVectorStore


# ---------------------------------------------------------------------
# DATACLASS
# ---------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------
# BLIP IMAGE CAPTIONING
# ---------------------------------------------------------------------

class BlipCaptioner:
    def __init__(self, device="cpu"):
        name = "Salesforce/blip-image-captioning-base"
        print("[BLIP] Initializing...")
        self.device = device
        self.processor = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name).to(device)

    def caption(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------------------

class RAGPipeline:

    def __init__(
        self,
        db_uri="lancedb",
        table_name="cern_demo",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        enable_blip=True,
        device="cpu",
    ):
        self.device = device
        self.embed_model = SentenceTransformer(embed_model_name)

        print("[Embedder] Loading SentenceTransformer:", embed_model_name)

        # LanceDB vector store
        self.store = LanceVectorStore(
            db_uri=db_uri,
            table_name=table_name,
            dim=384,
        )

        # BLIP
        self.captioner = BlipCaptioner(device) if enable_blip else None

    # -----------------------------------------------------------------
    # PDF TEXT EXTRACTION
    # -----------------------------------------------------------------

    def extract_text(self, pdf_path):
        print(f"[Extract] Text from {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        pages = [{"page": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]
        doc.close()
        return pages

    # -----------------------------------------------------------------
    # PDF IMAGE CAPTIONING
    # -----------------------------------------------------------------

    def extract_images_with_captions(self, pdf_path):
        if not self.captioner:
            return []

        print("[Extract] Images + captions")
        doc = fitz.open(pdf_path)
        caps = []

        for page_idx, page in enumerate(doc):
            images = page.get_images(full=True)
            for img_i, img in enumerate(images):
                xref = img[0]
                base = doc.extract_image(xref)
                pil_img = Image.open(BytesIO(base["image"])).convert("RGB")
                caption = self.captioner.caption(pil_img)

                caps.append({
                    "page": page_idx + 1,
                    "caption": caption,
                    "id": f"{page_idx+1}_{img_i+1}_{uuid.uuid4().hex[:8]}"
                })

        doc.close()
        print(f"[BLIP] Captions generated: {len(caps)}")
        return caps

    # -----------------------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------------------

    def chunk_pages(self, pages, image_caps, max_chars=800, overlap=150, source_name="document.pdf"):
        print("[Chunk] Chunking…")

        caps_by_page = {}
        for c in image_caps or []:
            caps_by_page.setdefault(c["page"], []).append(c["caption"])

        chunks = []

        for entry in pages:
            page = entry["page"]
            text = entry["text"].replace("\n", " ")

            fig_txt = ""
            for cap in caps_by_page.get(page, []):
                fig_txt += f"\n[FIGURE] {cap}"

            full = (text + fig_txt).strip()

            start = 0
            idx = 0

            while start < len(full):
                end = min(start + max_chars, len(full))
                chunk_text = full[start:end].strip()

                if len(chunk_text) > 30:
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            text=chunk_text,
                            source=source_name,
                            page=page,
                            chunk_index=idx,
                            metadata={"page": page, "chunk_index": idx},
                        )
                    )

                idx += 1
                if end >= len(full):
                    break
                start = end - overlap

        print(f"[Chunk] Total chunks: {len(chunks)}")
        return chunks

    # -----------------------------------------------------------------
    # EMBEDDING
    # -----------------------------------------------------------------

    def embed(self, texts):
        return np.array(self.embed_model.encode(texts, show_progress_bar=False), dtype="float32")

    # -----------------------------------------------------------------
    # INGESTION
    # -----------------------------------------------------------------

    def ingest_chunks(self, chunks):
        print("[Ingest] Storing in LanceDB…")

        texts = [c.text for c in chunks]
        vecs = self.embed(texts)

        rows = []
        for i, c in enumerate(chunks):
            rows.append({
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "vector": vecs[i],  # list of 384 floats
            })

        self.store.add(rows)
        print(f"[Ingest] Added {len(rows)} rows.")

    def ingest_pdf(self, pdf_path):
        name = os.path.basename(pdf_path)
        pages = self.extract_text(pdf_path)
        caps = self.extract_images_with_captions(pdf_path)
        chunks = self.chunk_pages(pages, caps, source_name=name)
        self.ingest_chunks(chunks)

    # -----------------------------------------------------------------
    # QUERY
    # -----------------------------------------------------------------

    def query(self, text, top_k=5):
        vec = self.embed([text])[0].tolist()
        rows = self.store.search(vec, top_k=top_k)

        return [
            {
                "score": r.get("_distance"),
                "text": r.get("text"),
                "source": r.get("source"),
                "page": r.get("page"),
                "chunk_index": r.get("chunk_index")
            }
            for r in rows
        ]
