import os
import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
GRAPHS_DIR = OUTPUTS / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

PDF_PATH = os.getenv("PDF_PATH")
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling-serve:5001")

print("[Graphs] Extracting graphs via Docling...")

with open(PDF_PATH, "rb") as f:
    resp = requests.post(f"{DOCLING_URL}/figures", files={"file": f})

resp.raise_for_status()
figures = resp.json().get("figures", [])

index = []
for fig in figures:
    filename = f"page_{fig['page']}_graph_{fig['index']}.png"
    out = GRAPHS_DIR / filename
    with open(out, "wb") as f:
        f.write(bytes(fig["data"]))
    index.append({"page": fig["page"], "image_path": str(out)})

with open(OUTPUTS / "graphs_index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"[Graphs] Saved {len(index)} graphs → {GRAPHS_DIR}")
