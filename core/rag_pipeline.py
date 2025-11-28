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
        figure_root="static/figures",
    ):
        self.device = device
        self.embed_model = SentenceTransformer(embed_model_name)

        print("[Embedder] Loading SentenceTransformer:", embed_model_name)

        self.store = LanceVectorStore(
            db_uri=db_uri,
            table_name=table_name,
            dim=384,
        )

        self.captioner = BlipCaptioner(device) if enable_blip else None
        self.figure_root = figure_root

    # -----------------------------------------------------------------
    # PDF TEXT EXTRACTION
    # -----------------------------------------------------------------

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        print(f"[Extract] Text from {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        pages = [{"page": i + 1, "text": p.get_text("text")} for i, p in enumerate(doc)]
        doc.close()
        return pages

    # -----------------------------------------------------------------
    # IMAGE FILTERS (REMOVE PAGE PREVIEWS)
    # -----------------------------------------------------------------

    def _should_reject_image(self, page_pix, img_pil):
        pw, ph = page_pix.width, page_pix.height
        w, h = img_pil.size

        # Reject images >60% of page area (page previews)
        if (w * h) > 0.60 * (pw * ph):
            return True

        # Reject images that have A4-like ratio (previews)
        ratio = w / h
        if abs(ratio - 0.70) < 0.05 or abs(ratio - 1.414) < 0.05:
            return True

        return False

    # -----------------------------------------------------------------
    # PDF IMAGE CAPTIONING + FILTERING + SAVING
    # -----------------------------------------------------------------

    def extract_images_with_captions(self, pdf_path: str) -> List[Dict[str, Any]]:
        if not self.captioner:
            return []

        print("[Extract] Images + captions (with filtering)")
        doc = fitz.open(pdf_path)
        caps = []

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_fig_root = os.path.join(self.figure_root, pdf_name)
        os.makedirs(pdf_fig_root, exist_ok=True)

        for page_idx, page in enumerate(doc):
            images = page.get_images(full=True)
            page_pix = page.get_pixmap()

            for img_i, img in enumerate(images):
                xref = img[0]
                base = doc.extract_image(xref)
                pil_img = Image.open(BytesIO(base["image"])).convert("RGB")

                # FILTER 1: remove page previews
                if self._should_reject_image(page_pix, pil_img):
                    continue

                # Caption
                caption = self.captioner.caption(pil_img)

                # FILTER 2: remove bad captions junk
                junk_prefixes = ["a page", "the cover", "a white sheet", "page "]
                if any(caption.lower().startswith(j) for j in junk_prefixes):
                    continue

                # Save figure
                short_id = uuid.uuid4().hex[:8]
                filename = f"page{page_idx+1}_img{img_i+1}_{short_id}.png"
                image_path = os.path.join(pdf_fig_root, filename)
                pil_img.save(image_path)

                caps.append({
                    "page": page_idx + 1,
                    "caption": caption,
                    "image_path": image_path,
                    "id": f"{page_idx+1}_{img_i+1}_{short_id}",
                })

        doc.close()
        print(f"[BLIP] Filtered figures: {len(caps)}")
        return caps

    # -----------------------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------------------

    def chunk_pages(
        self,
        pages,
        image_caps,
        max_chars=800,
        overlap=150,
        source_name="document.pdf",
    ):
        print("[Chunk] Chunking…")

        caps_by_page = {}
        for c in image_caps or []:
            caps_by_page.setdefault(c["page"], []).append(c)

        chunks = []

        for entry in pages:
            page = entry["page"]
            text = entry["text"].replace("\n", " ")

            figs = caps_by_page.get(page, [])
            fig_captions = [c["caption"] for c in figs]
            fig_paths = [c["image_path"] for c in figs]

            fig_txt = ""
            for cap in fig_captions:
                fig_txt += f"\n[FIGURE] {cap}"

            full = (text + fig_txt).strip()

            start = 0
            idx = 0

            while start < len(full):
                end = min(start + max_chars, len(full))
                chunk_text = full[start:end].strip()

                if len(chunk_text) > 30:
                    metadata = {
                        "page": page,
                        "chunk_index": idx,
                        "figure_captions": fig_captions,
                        "figure_paths": fig_paths,
                        "has_figures": len(fig_captions) > 0,
                    }

                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            text=chunk_text,
                            source=source_name,
                            page=page,
                            chunk_index=idx,
                            metadata=metadata,
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

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embed_model.encode(texts, show_progress_bar=False), dtype="float32")

    # -----------------------------------------------------------------
    # INGESTION
    # -----------------------------------------------------------------

    def ingest_chunks(self, chunks: List[Chunk]):
        print("[Ingest] Storing in LanceDB…")

        texts = [c.text for c in chunks]
        vecs = self.embed(texts)

        rows = []
        for i, c in enumerate(chunks):
            meta = c.metadata or {}
            rows.append({
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "vector": vecs[i],
                "figure_captions": meta.get("figure_captions", []),
                "figure_paths": meta.get("figure_paths", []),
                "has_figures": meta.get("has_figures", False),
            })

        self.store.add(rows)
        print(f"[Ingest] Added {len(rows)} rows.")

    def ingest_pdf(self, pdf_path: str):
        name = os.path.basename(pdf_path)
        print(f"[Ingest] PDF: {name}")
        pages = self.extract_text(pdf_path)
        caps = self.extract_images_with_captions(pdf_path)
        chunks = self.chunk_pages(pages, caps, source_name=name)
        self.ingest_chunks(chunks)

    def ingest_folder(self, folder_path: str, recursive: bool = False):
        print(f"[Ingest] Scanning folder: {folder_path}")
        folder_path = os.path.abspath(folder_path)

        for root, dirs, files in os.walk(folder_path):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    self.ingest_pdf(os.path.join(root, fn))

            if not recursive:
                break

    # -----------------------------------------------------------------
    # SIMILARITY
    # -----------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    # -----------------------------------------------------------------
    # HYBRID QUERY (TEXT + FIGURES)
    # -----------------------------------------------------------------

    def query(self, text: str, top_k: int = 5, oversample: int = 3):
        alpha = 0.7

        q_vec = self.embed([text])[0]

        initial_k = max(top_k * oversample, top_k)
        rows = self.store.search(q_vec.tolist(), top_k=initial_k)

        results = []

        for r in rows:
            text_vec = np.array(r.get("vector"), dtype="float32")
            text_score = self._cosine_sim(q_vec, text_vec)

            fig_caps = r.get("figure_captions") or []

            # caption score
            if fig_caps:
                cap_vec = self.embed([" ".join(fig_caps)])[0]
                caption_score = self._cosine_sim(q_vec, cap_vec)
            else:
                caption_score = 0.0

            hybrid_score = alpha * text_score + (1 - alpha) * caption_score

            label = "Top Figure Mentions" if fig_caps else "Top Text Sections"

            results.append({
                "score": hybrid_score,
                "text_score": text_score,
                "caption_score": caption_score,
                "label": label,
                "text": r["text"],
                "source": r["source"],
                "page": r["page"],
                "chunk_index": r["chunk_index"],
                "figure_captions": fig_caps,
                "figure_paths": r.get("figure_paths", []),
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    # -----------------------------------------------------------------
    # LLM SUMMARISATION PROMPT
    # -----------------------------------------------------------------

    def build_summary_prompt(self, question, results, max_chunks=8):
        selected = results[:max_chunks]

        blocks = []
        for i, r in enumerate(selected, start=1):
            fig_caps = r.get("figure_captions") or []
            fig_str = "; ".join(fig_caps) if fig_caps else "None"

            blocks.append(
                f"### Chunk {i} (page {r['page']}, {r['label']})\n"
                f"Text:\n{r['text']}\n\n"
                f"Figure captions: {fig_str}\n"
            )

        context = "\n\n".join(blocks)

        return f"""You are an assistant helping with scientific PDFs.

User question:
{question}

Context:
{context}

Write a concise answer based ONLY on the context. Focus on figures if the question involves figures.
"""


