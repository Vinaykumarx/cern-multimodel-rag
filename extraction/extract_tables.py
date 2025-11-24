import os
import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS / "tables"
TABLES_DIR.mkdir(exist_ok=True)

PDF_PATH = os.getenv("PDF_PATH")
DOCLING_URL = os.getenv("DOCLING_URL", "http://docling-serve:5001")

print("[Tables] Extracting tables via Docling...")

with open(PDF_PATH, "rb") as f:
    resp = requests.post(f"{DOCLING_URL}/tables", files={"file": f})

resp.raise_for_status()
data = resp.json()

tables = data.get("tables", [])
index = []

for t in tables:
    filename = f"page_{t['page']}_table_{t['index']}.json"
    out = TABLES_DIR / filename
    with open(out, "w") as f:
        json.dump(t["cells"], f, indent=2)
    index.append({"page": t["page"], "table_json": str(out)})

with open(OUTPUTS / "tables_index.json", "w") as f:
    json.dump(index, f, indent=2)

print(f"[Tables] Saved {len(index)} tables → {TABLES_DIR}")
