# core/pdf_loader.py

from typing import List, Dict, Any
from pathlib import Path
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image


def extract_text_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract plain text per page using PyMuPDF.
    Returns: [{"page": int, "text": str}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def extract_page_images(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all images from the PDF as PIL.Image instances.
    Returns: [{"page": int, "image": PIL.Image, "xref": int}, ...]
    """
    pdf_path = str(Path(pdf_path))
    doc = fitz.open(pdf_path)
    out = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)  # list of tuples
        for img_info in images:
            xref = img_info[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            out.append(
                {
                    "page": page_index + 1,
                    "image": pil_img,
                    "xref": xref,
                }
            )

    doc.close()
    return out
