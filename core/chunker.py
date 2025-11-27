# core/chunker.py

from dataclasses import dataclass
from typing import List, Dict, Any
import uuid


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    metadata: Dict[str, Any]


def chunk_pages(
    pages: List[Dict[str, Any]],
    image_caps: List[Dict[str, Any]],
    max_chars: int = 800,
    overlap: int = 150,
    source_name: str = "document.pdf",
) -> List[Chunk]:
    """
    Chunk page texts and append associated figure captions.
    `pages`: [{"page": int, "text": str}, ...]
    `image_caps`: [{"page": int, "caption": str, "image_id": str}, ...]
    """
    caps_by_page: Dict[int, List[str]] = {}
    for c in image_caps:
        page = int(c["page"])
        caps_by_page.setdefault(page, []).append(c["caption"])

    chunks: List[Chunk] = []

    for entry in pages:
        page_num = int(entry["page"])
        raw_text = (entry.get("text") or "").replace("\n", " ")

        fig_text = ""
        for cap in caps_by_page.get(page_num, []):
            fig_text += f"\n[FIGURE] {cap}"

        full_text = (raw_text + fig_text).strip()
        if not full_text:
            continue

        start = 0
        idx = 0
        n = len(full_text)

        while start < n:
            end = min(start + max_chars, n)
            chunk_text = full_text[start:end].strip()

            # Avoid ultra tiny chunks
            if len(chunk_text) > 30:
                cid = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        id=cid,
                        text=chunk_text,
                        source=source_name,
                        page=page_num,
                        chunk_index=idx,
                        metadata={"page": page_num, "chunk_index": idx},
                    )
                )

            idx += 1
            if end >= n:
                break
            start = end - overlap

    print(f"[Chunk] Total chunks: {len(chunks)}")
    return chunks
