import json
from pathlib import Path

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "outputs"
IMAGES_DIR = OUT_DIR / "images"
GRAPHS_DIR = OUT_DIR / "graphs"
FIGURES_JSON = OUT_DIR / "figures_index.json"

MODEL_NAME = "Salesforce/blip-image-captioning-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[BLIP] Loading {MODEL_NAME} on {device} ...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def infer_page_from_filename(path: Path):
    """
    Try to infer page number from patterns like:
    - page_8_img_46.png
    - page_12_graph_1.png
    """
    stem = path.stem
    parts = stem.split("_")
    for i in range(len(parts)):
        if parts[i] == "page" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def generate_caption(image_path: Path, kind: str) -> str:
    """
    Generate a caption with a small hint based on kind.
    kind ∈ {"image", "graph"}
    """
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[BLIP] Error opening image {image_path}: {e}")
        return ""

    if kind == "graph":
        prompt = "A scientific graph or plot from a CERN materials radiation report. Describe briefly."
    else:
        prompt = "A scientific figure from a CERN materials radiation report. Describe briefly."

    inputs = processor(raw_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.strip()


def main():
    figures = []

    # 1) Normal extracted images (logos, small diagrams, etc.)
    if IMAGES_DIR.exists():
        image_paths = sorted(IMAGES_DIR.glob("*.png"))
        print(f"[BLIP] Found {len(image_paths)} images in {IMAGES_DIR}")
        for img_path in image_paths:
            print(f"[BLIP] Captioning image {img_path.name} ...")
            caption = generate_caption(img_path, kind="image")
            page = infer_page_from_filename(img_path)
            figures.append({
                "image_path": str(img_path),
                "caption": caption,
                "page": page,
                "kind": "image"
            })

    # 2) Graph candidates from rendered pages
    if GRAPHS_DIR.exists():
        graph_paths = sorted(GRAPHS_DIR.glob("*.png"))
        print(f"[BLIP] Found {len(graph_paths)} graphs in {GRAPHS_DIR}")
        for g_path in graph_paths:
            print(f"[BLIP] Captioning graph {g_path.name} ...")
            caption = generate_caption(g_path, kind="graph")
            page = infer_page_from_filename(g_path)
            figures.append({
                "image_path": str(g_path),
                "caption": caption,
                "page": page,
                "kind": "graph"
            })

    with open(FIGURES_JSON, "w") as f:
        json.dump(figures, f, indent=2)

    print(f"[BLIP] Saved {len(figures)} figure/graph captions to {FIGURES_JSON}")


if __name__ == "__main__":
    main()
