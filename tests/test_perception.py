import json
import types
from pathlib import Path

import PIL.Image as Image
import pytest


def test_ocr_run_and_save(monkeypatch, tmp_img, tmp_path):
    # Mock pytesseract and easyocr
    import src.perception.ocr as ocr

    def fake_tess(img):
        return "detected text"

    class FakeEasy:
        def __init__(self, langs, gpu=True): pass
        def readtext(self, p, detail=0): return ["easy ocr text"]

    monkeypatch.setattr(ocr, "pytesseract", types.SimpleNamespace(image_to_string=fake_tess))
    monkeypatch.setattr(ocr, "easyocr", types.SimpleNamespace(Reader=FakeEasy))
    # Force EASY reader
    ocr._EASY = FakeEasy(["en"], gpu=False)

    res = ocr.run_ocr([tmp_img])
    assert res and res[0]["ocr"]

    ocr.save_ocr("rec1", res, str(tmp_path))
    out = tmp_path / "rec1_ocr.json"
    assert out.exists()
    js = json.loads(out.read_text())
    assert js[0]["path"].endswith("img.jpg")


def test_keyframes_extract(monkeypatch, tmp_path):
    # Create a fake ffmpeg chain that creates files
    import src.perception.keyframes as kf

    class FakeChain:
        def __init__(self, out_dir):
            self.out_dir = out_dir
        def filter(self, *a, **kw): return self
        def output(self, *a, **kw): return self
        def overwrite_output(self): return self
        def run(self, quiet=True):
            # simulate frame outputs
            for i in range(3):
                p = Path(self.out_dir) / f"frame_{i:06d}.jpg"
                Image.new("RGB", (8, 8)).save(p)

    def fake_input(video_path):
        return FakeChain(str(tmp_path))

    monkeypatch.setattr(kf, "ffmpeg", types.SimpleNamespace(input=fake_input))
    out_files = kf.extract_keyframes("vid.mp4", str(tmp_path), fps=1.0)
    assert len(out_files) == 3
    assert Path(out_files[0]).name.startswith("frame_")


def test_asr_run_and_save(monkeypatch, tmp_path):
    import src.perception.asr as asr

    class FakeModel:
        def transcribe(self, wav, beam_size=5):
            class Info: language = "en"
            segs = [types.SimpleNamespace(start=0.0, end=1.0, text="hi there")]
            return segs, Info()

    def fake_extract_audio(video_path, wav_path, sr=16000):
        # create empty wav
        Path(wav_path).write_bytes(b"\x00\x00")

    monkeypatch.setattr(asr, "WhisperModel", lambda *a, **k: FakeModel())
    monkeypatch.setattr(asr, "extract_audio", fake_extract_audio)

    out = asr.run_asr("foo.mp4")
    assert out["language"] == "en" and "hi there" in out["text"]

    asr.save_asr("rid", out, str(tmp_path))
    assert (tmp_path / "rid_asr.json").exists()


def test_caption_images(monkeypatch, tmp_img):
    import src.perception.caption as cap

    class FakePipe:
        def __call__(self, img, max_new_tokens=40):
            return [{"generated_text": "a small green square"}]

    monkeypatch.setattr(cap, "pipeline", lambda *a, **k: FakePipe())
    # replace global pipeline instance
    cap._captioner = FakePipe()

    res = cap.caption_images([tmp_img])
    assert res[0]["caption"] == "a small green square"


def test_translate(monkeypatch):
    import src.perception.translate as tr

    class FakePipe:
        def __call__(self, texts, max_length=512):
            return [{"translation_text": t + " (en)"} for t in texts]

    monkeypatch.setattr(tr, "pipeline", lambda *a, **k: FakePipe())
    tr._translator = FakePipe()
    outs = tr.to_english(["hola", "mundo"])
    assert outs == ["hola (en)", "mundo (en)"]


def test_pack_text(tmp_path):
    import src.perception.pack_text as pk
    txt = pk.pack_text(str(tmp_path), "rid", "post", [{"path":"a.jpg","ocr":"A"}], {"text":"B"}, [{"path":"a.jpg","caption":"C"}])
    assert "=== POST_TEXT ===" in Path(txt).read_text()
