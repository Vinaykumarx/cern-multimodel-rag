import os
import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

PDF_PATH = os.getenv("PDF_PATH")
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling-serve:5001")

PAGES_TEXT_PATH = OUTPUTS / "pages_text.json"

if not PDF_PATH or not Path(PDF_PATH).exists():
    raise FileNotFoundError("❌ PDF_PATH missing or invalid.")

print("[Text] Sending PDF to Docling for analysis...")

with open(PDF_PATH, "rb") as f:
    files = {"file": f}
    resp = requests.post(f"{DOCLING_URL}/extract", files=files)

resp.raise_for_status()
data = resp.json()

pages = []

for i, page in enumerate(data.get("pages", [])):
    pages.append({
        "page": i + 1,
        "text": page.get("text", ""),
    })

with open(PAGES_TEXT_PATH, "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2)

print(f"[Text] Saved extracted text → {PAGES_TEXT_PATH}")
