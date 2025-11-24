import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

metadata = {}

files = [
    "pages_text.json",
    "images_index.json",
    "graphs_index.json",
    "figures_index.json",
    "tables_index.json"
]

for name in files:
    path = OUTPUTS / name
    if path.exists():
        with open(path) as f:
            metadata[name.replace(".json", "")] = json.load(f)

with open(OUTPUTS / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("[Metadata] metadata.json built successfully.")
