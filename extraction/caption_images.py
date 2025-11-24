import os
import json
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

IMAGES = OUTPUTS / "images_index.json"
GRAPHS = OUTPUTS / "graphs_index.json"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption(path):
    img = Image.open(path)
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def caption_file(index_path):
    if not Path(index_path).exists():
        return []

    with open(index_path) as f:
        items = json.load(f)

    results = []
    for item in items:
        if not Path(item["image_path"]).exists():
            continue
        cap = caption(item["image_path"])
        item["caption"] = cap
        results.append(item)

    return results


all_captions = caption_file(IMAGES) + caption_file(GRAPHS)

with open(OUTPUTS / "figures_index.json", "w") as f:
    json.dump(all_captions, f, indent=2)

print(f"[Caption] Wrote captions → {OUTPUTS/'figures_index.json'}")
