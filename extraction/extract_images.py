import os
import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
IMAGES_DIR = OUTPUTS / "images"
IMAGES_DIR.mkdir(exist_ok=True)

PDF_PATH = os.getenv("PDF_PATH")
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling-serve:5001")

print("[Images] Extracting images via Docling...")

with open(PDF_PATH, "rb") as f:
    resp = requests.post(f"{DOCLING_URL}/images", files={"file": f})

resp.raise_for_status()
images = resp.json().get("images", [])

saved = []

for img in images:
    filename = f"page_{img['page']}_img_{img['index']}.png"
    out = IMAGES_DIR / filename
    with open(out, "wb") as f:
        f.write(bytes(img["data"]))
    saved.append({"page": img["page"], "image_path": str(out)})

with open(OUTPUTS / "images_index.json", "w") as f:
    json.dump(saved, f, indent=2)

print(f"[Images] Saved {len(saved)} images → {IMAGES_DIR}")
