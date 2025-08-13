import os, json, tempfile, ffmpeg
from faster_whisper import WhisperModel

_ASR = None

def _get_asr_model(device="cuda", compute_type="float16"):
    global _ASR
    if _ASR is None:
        _ASR = WhisperModel("medium", device=device, compute_type=compute_type)
    return _ASR

def extract_audio(video_path: str, wav_path: str, sr: int = 16000):
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, acodec="pcm_s16le", ac=1, ar=str(sr))
        .overwrite_output()
        .run(quiet=True)
    )

def run_asr(video_path: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        extract_audio(video_path, wav)
        model = _get_asr_model()
        segments, info = model.transcribe(wav, beam_size=5)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {"language": info.language, "segments": segs, "text": " ".join(s["text"] for s in segs)}

def save_asr(rec_id: str, asr: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{rec_id}_asr.json"), "w", encoding="utf-8") as f:
        json.dump(asr, f, ensure_ascii=False, indent=2)
