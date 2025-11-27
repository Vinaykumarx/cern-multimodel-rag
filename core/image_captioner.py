# core/image_captioner.py

from typing import Optional

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


class BlipCaptioner:
    """
    Thin wrapper around BLIP image captioning model.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
    ):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    def caption(self, img: Image.Image, max_new_tokens: int = 30) -> str:
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text.strip()


def maybe_init_blip(enable_blip: bool, device: str = "cpu") -> Optional[BlipCaptioner]:
    if not enable_blip:
        return None

    try:
        print("[BLIP] Initializing...")
        return BlipCaptioner(device=device)
    except Exception as e:
        print(f"[BLIP] Failed to initialize BLIP: {e}")
        return None
