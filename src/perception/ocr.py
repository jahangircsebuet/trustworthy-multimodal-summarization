import os, json
from typing import List
from PIL import Image
import pytesseract

try:
    import easyocr
    _EASY = easyocr.Reader(["en"], gpu=True)
except Exception:
    _EASY = None

def run_ocr(image_paths: List[str]) -> List[dict]:
    out = []
    for p in image_paths:
        text = ""
        try:
            text = pytesseract.image_to_string(Image.open(p))
        except Exception:
            pass
        if not text and _EASY:
            try:
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
