import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent.parent
PDF_ENV = os.getenv('PDF_PATH')
if PDF_ENV:
    pdf_path = Path(PDF_ENV)
else:
    pdf_path = BASE / 'data' / 'CERN_Yellow_Report_357576.pdf'
out_dir = BASE / "outputs"
graphs_dir = out_dir / "graphs"
graphs_dir.mkdir(parents=True, exist_ok=True)

def extract_graph_candidates_from_page(page_image: np.ndarray):
    """
    Simple Option A:
    - convert to gray
    - edge detect
    - find largest rectangular contour
    - return its bounding box if it looks big enough
    """
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    h, w = gray.shape
    candidates = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 0.05 * w * h:
            # too small
            continue
        if area > 0.9 * w * h:
            # almost full page, likely background
            continue

        aspect = cw / float(ch)
        if aspect < 0.5 or aspect > 3.0:
            # very tall or very long; skip for simple mode
            continue

        candidates.append((x, y, cw, ch))
        # For Option A: take only the largest suitable one
        break

    return candidates

def main():
    doc = fitz.open(str(pdf_path))
    print(f"[GRAPHS] Opened {pdf_path}, pages={len(doc)}")

    count = 0

    for page_index, page in enumerate(doc):
        # Render page at moderate DPI
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        # If alpha channel present, drop it
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        candidates = extract_graph_candidates_from_page(img)
        if not candidates:
            continue

        for ci, (x, y, cw, ch) in enumerate(candidates):
            crop = img[y:y + ch, x:x + cw]
            out_path = graphs_dir / f"page_{page_index+1}_graph_{ci+1}.png"
            cv2.imwrite(str(out_path), crop)
            count += 1

    print(f"[GRAPHS] Saved {count} graph candidate images into {graphs_dir}")

if __name__ == "__main__":
    main()