from typing import List, Dict
from PIL import Image
from transformers import pipeline

# BLIP2 is heavy; swap to a lighter BLIP captioner if you prefer.
_captioner = pipeline("image-to-text", model="Salesforce/blip2-flan-t5-xl", device_map="auto")

from typing import List, Dict
from io import BytesIO
import re
import httpx
from PIL import Image, ImageOps

# Simple URL detector
_IS_URL = re.compile(r"^https?://", re.I)

def _load_image(path_or_url: str, timeout: int = 20) -> Image.Image:
    """
    Loads a PIL Image from a local path or an HTTP(S) URL.
    - Follows redirects (needed for picsum/photos, etc.)
    - Applies EXIF orientation and converts to RGB
    - Optionally downscales very large images to keep memory reasonable
    """
    if _IS_URL.match(path_or_url or ""):
        # Fetch bytes over HTTP
        headers = {
            "User-Agent": "Mozilla/5.0 (captioner/1.0)",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        with httpx.Client(follow_redirects=True, headers=headers, timeout=timeout) as client:
            resp = client.get(path_or_url)
            resp.raise_for_status()
            data = resp.content
        img = Image.open(BytesIO(data))
    else:
        img = Image.open(path_or_url)

    # Normalize orientation + color mode
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Optional: downscale very large images (helps VRAM/latency)
    # Comment out if you prefer full-res
    max_side = 1600
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    return img

def caption_images(image_paths: List[str], max_new_tokens: int = 40) -> List[Dict]:
    """
    Works for both local paths and HTTP(S) URLs.
    Returns: [{"path": p, "caption": cap}]
    """
    print("caption_images called...")
    out = []
    for p in image_paths:
        try:
            print("p:", p)
            img = _load_image(p)
            # _captioner should be your already-initialized image captioning pipeline/model
            res = _captioner(img, max_new_tokens=max_new_tokens)
            # Some captioners return different keys; adjust if needed
            cap = (res[0].get("generated_text") or res[0].get("caption") or "").strip()
        except Exception as e:
            print(f"[warn] caption failed for {p}: {e}")
            cap = ""
        item = {"path": p, "caption": cap}
        out.append(item)
        print("caption:", item)
    return out


if __name__ == "__main__":
    captions = caption_images(["/home/malam10/projects/trustworthy-multimodal-summarization/img/image.png"])
    for cap in captions:
        print(cap["path"], cap["caption"])
