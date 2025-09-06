from typing import List, Dict
from PIL import Image
from transformers import pipeline
import requests
from io import BytesIO

# BLIP2 is heavy; swap to a lighter BLIP captioner if you prefer.
_captioner = pipeline("image-to-text", model="Salesforce/blip2-flan-t5-xl", device_map="auto")

def load_image(path: str):
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(path)

def caption_images(image_paths: List[str], max_new_tokens=40) -> List[Dict]:
    out = []
    for p in image_paths:
        try:
            img = load_image(p)
            res = _captioner(img, max_new_tokens=max_new_tokens)
            cap = res[0]["generated_text"].strip()
        except Exception as e:
            cap = ""
        out.append({"path": p, "caption": cap})
    return out