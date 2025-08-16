# tests/test_pack_text_integration.py
import importlib
from pathlib import Path
import sys

# Ensure the project root (the dir that contains "src/") is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_pack_text_creates_textbag(tmp_path, monkeypatch):
    # Import your actual module by package path
    mod = importlib.import_module("src.perception.pack_text")

    # If pack_text uses SECTION(), patch it so the header is predictable for assertions
    monkeypatch.setattr(mod, "SECTION", lambda s: f"## {s} ##", raising=False)

    rec_id = "rec123"
    post_text = "Original post text goes here."
    translations = "Translated English text."
    ocr_items = [
        {"path": "img1.jpg", "ocr": "Stop sign ahead"},
        {"path": "img2.jpg", "ocr": "Main Street"},
    ]
    asr_obj = {"text": "Hello and welcome to the event."}
    captions = [
        {"path": "img1.jpg", "caption": "A person holding a trophy"},
        {"path": "img2.jpg", "caption": "Crowd at night"},
    ]

    textbag_path = mod.pack_text(
        out_dir=str(tmp_path),
        rec_id=rec_id,
        post_text=post_text,
        ocr_items=ocr_items,
        asr_obj=asr_obj,
        captions=captions,
        translations=translations,
    )

    tb = Path(textbag_path)
    assert tb.exists()
    assert tb.name == f"{rec_id}_textbag.txt"

    content = tb.read_text(encoding="utf-8")
    # Headers present (and in order)
    assert "## POST_TEXT ##" in content
    assert "## TRANSLATION_EN ##" in content
    assert "## OCR ##" in content
    assert "## ASR ##" in content
    assert "## CAPTIONS ##" in content
    pos = {h: content.index(h) for h in [
        "## POST_TEXT ##", "## TRANSLATION_EN ##", "## OCR ##", "## ASR ##", "## CAPTIONS ##"
    ]}
    assert pos["## POST_TEXT ##"] < pos["## TRANSLATION_EN ##"] < pos["## OCR ##"] < pos["## ASR ##"] < pos["## CAPTIONS ##"]

    # Enumerated items + body text
    assert "[1] img1.jpg: Stop sign ahead" in content
    assert "[2] img2.jpg: Main Street" in content
    assert "[1] img1.jpg: A person holding a trophy" in content
    assert "[2] img2.jpg: Crowd at night" in content
    assert "Original post text goes here." in content
    assert "Translated English text." in content
    assert "Hello and welcome to the event." in content
