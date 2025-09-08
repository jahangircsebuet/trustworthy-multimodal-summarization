import os, json
from typing import List
from PIL import Image
import pytesseract
import requests
from io import BytesIO

try:
    import easyocr
    _EASY = easyocr.Reader(["en"], gpu=True)
except Exception:
    _EASY = None

def load_image(path: str):
    if path.startswith("http://") or path.startswith("https://"):
        response = requests.get(path)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(path)

def run_ocr(image_paths: List[str]) -> List[dict]:
    out = []
    for p in image_paths:
        text = ""
        try:
            img = load_image(p)
            text = pytesseract.image_to_string(img)
        except Exception:
            pass
        if not text and _EASY:
            try:
                if p.startswith("http://") or p.startswith("https://"):
                    img = load_image(p)
                    img.save("tmp_ocr_img.jpg")
                    res = _EASY.readtext("tmp_ocr_img.jpg", detail=0)
                    os.remove("tmp_ocr_img.jpg")
                else:
                    res = _EASY.readtext(p, detail=0)
                text = "\n".join(res)
            except Exception:
                pass
        out.append({"path": p, "ocr": text.strip()})
    return out

def save_ocr(rec_id: str, results: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{rec_id}_ocr.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)