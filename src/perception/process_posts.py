# tools/run_pack_textbags.py
from __future__ import annotations
from pathlib import Path
import json
import sys
from typing import Any, Dict, List, Union

# --- Make sure "src/" is importable (assumes this script lives at project_root/tools/) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports you already have ---
from src.perception.pack_text import pack_text

# ⬇️ Adjust these to your actual module paths
from src.perception.caption import caption_images
from src.perception.ocr import run_ocr
from src.perception.asr import run_asr
from src.perception.translate import to_english


# --- The process_posts function from earlier (import if you already saved it) ---
from typing import Callable, Optional

def process_posts(
    posts: List[Dict[str, Any]],
    out_dir: str,
    caption_fn: Callable[[List[str], int], List[Dict[str, str]]],
    ocr_fn: Callable[[List[str]], List[Dict[str, str]]],
    asr_fn: Callable[[str], Dict[str, Any]],
    translate_fn: Optional[Callable[[List[str]], List[str]]] = None,
    translate_non_en: bool = True,
    caption_max_new_tokens: int = 40,
) -> List[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_paths: List[str] = []

    for post in posts:
        post_id = post["id"]
        post_text = (post.get("text") or "").strip()
        images: List[str] = post.get("images") or []
        video = post.get("video")
        lang = (post.get("lang") or "en").lower()

        
        ocr_items = []
        asr_obj = []
        

        # captions
        captions: List[Dict[str, str]] = caption_fn(images, max_new_tokens=caption_max_new_tokens) if images else []

        print("process_posts.captions: ", captions)
        
        # ocr
        # ocr_items: List[Dict[str, str]] = ocr_fn(images) if images else []
        # asr
        # asr_obj = asr_fn(video) if video else None
        # translation (post text only)
        translation_en = None
        # if translate_non_en and translate_fn and lang != "en" and post_text:
        #     try:
        #         translation_en = translate_fn([post_text])[0]
        #     except Exception as e:
        #         print(f"[warn] translation failed for {post_id}: {e}")

        # pack textbag
        textbag_path = pack_text(
            out_dir=out_dir,
            rec_id=post_id,
            post_text=post_text,
            ocr_items=ocr_items,
            asr_obj=asr_obj,
            captions=captions,
            translations=translation_en,
        )
        out_paths.append(textbag_path)
        print(f"[ok] packed {post_id} -> {textbag_path}")

    return out_paths


def load_posts(dataset_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads posts from:
      - a JSON list, or
      - a dict with key 'posts', or
      - JSONL (one post per line)
    """
    p = Path(dataset_path)
    text = p.read_text(encoding="utf-8").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "posts" in obj:
            return obj["posts"]
        raise ValueError("Unsupported JSON structure")
    except Exception:
        # Try JSONL
        posts: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                posts.append(json.loads(line))
        return posts


def main():
    dataset_path = "/home/malam10/projects/trustworthy-multimodal-summarization/datasets/dataset.json"
    out_dir = str(PROJECT_ROOT / "outputs" / "textbags")

    posts = load_posts(dataset_path)
    print(f"[info] loaded {len(posts)} posts from {dataset_path}")

    textbags = process_posts(
        posts=posts,
        out_dir=out_dir,
        caption_fn=caption_images,
        ocr_fn=run_ocr,
        asr_fn=run_asr,
        translate_fn=to_english,      # set to None to disable translation
        translate_non_en=True,
        caption_max_new_tokens=40,
    )
    print(f"[done] wrote {len(textbags)} textbags to: {out_dir}")

    for tb in textbags:
        print("tb:", tb)


if __name__ == "__main__":
    main()
