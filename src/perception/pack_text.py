import os

SECTION = lambda x: f"\n\n=== {x} ===\n"

def pack_text(out_dir: str, rec_id: str, post_text: str, ocr_items, asr_obj, captions, translations=None) -> str:
    parts = [SECTION("POST_TEXT"), post_text.strip()]
    if translations:
        parts += [SECTION("TRANSLATION_EN"), translations.strip()]
    if ocr_items:
        parts += [SECTION("OCR"), "\n".join(f"[{i+1}] {o['path']}: {o['ocr']}" for i,o in enumerate(ocr_items))]
    if asr_obj and asr_obj.get("text"):
        parts += [SECTION("ASR"), asr_obj["text"]]
    if captions:
        parts += [SECTION("CAPTIONS"), "\n".join(f"[{i+1}] {c['path']}: {c['caption']}" for i,c in enumerate(captions))]
    txt = "\n".join(parts).strip()
    os.makedirs(out_dir, exist_ok=True)
    textbag = os.path.join(out_dir, f"{rec_id}_textbag.txt")
    with open(textbag, "w", encoding="utf-8") as f:
        f.write(txt)
    return textbag
