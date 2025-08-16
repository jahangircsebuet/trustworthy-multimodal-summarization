# src/perception/asr.py
import os, json, tempfile, argparse, shutil, subprocess
from urllib.parse import urlparse

import requests
from faster_whisper import WhisperModel

_ASR = None

def _get_asr_model(device="cuda", compute_type="float16"):
    global _ASR
    if _ASR is None:
        _ASR = WhisperModel("medium", device=device, compute_type=compute_type)
    return _ASR

def extract_audio(video_path: str, wav_path: str, sr: int = 16000):
    """
    Use the system 'ffmpeg' binary directly (no ffmpeg-python import).
    """
    cmd = [
        "ffmpeg",
        "-y",                # overwrite output
        "-i", video_path,    # input video
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sr),
        wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr.decode('utf-8', errors='ignore')}")

def run_asr(video_path: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        extract_audio(video_path, wav)
        model = _get_asr_model()
        segments, info = model.transcribe(wav, beam_size=5)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {
            "language": getattr(info, "language", "unknown"),
            "segments": segs,
            "text": " ".join(s["text"] for s in segs).strip()
        }

def save_asr(rec_id: str, asr: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{rec_id}_asr.json"), "w", encoding="utf-8") as f:
        json.dump(asr, f, ensure_ascii=False, indent=2)

# -----------------------------
# JSON / URL helpers
# -----------------------------

def _is_url(x: str) -> bool:
    try:
        return urlparse(x).scheme in ("http", "https")
    except Exception:
        return False

def _download_to_tmp(url: str) -> str:
    clean = url.split("?")[0].split("#")[0]
    base = os.path.basename(clean)
    suffix = "." + base.split(".")[-1] if "." in base else ""
    fd, tmp = tempfile.mkstemp(suffix=suffix); os.close(fd)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
    return tmp

def _process_single(video_path_or_url: str) -> dict:
    local = video_path_or_url
    tmp = None
    try:
        if _is_url(video_path_or_url):
            local = _download_to_tmp(video_path_or_url)
            tmp = local
        return run_asr(local)
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

def process_posts(posts_json: str, out_dir: str):
    with open(posts_json, "r", encoding="utf-8") as f:
        posts = json.load(f)

    for post in posts:
        rec_id = post.get("id") or "unknown"
        video_url = post.get("video")
        if not video_url:
            continue
        print(f"[ASR] Processing {rec_id} from {video_url}")
        try:
            asr_result = _process_single(video_url)
            save_asr(rec_id, asr_result, out_dir)
        except Exception as e:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{rec_id}_asr.error.txt"), "w", encoding="utf-8") as ef:
                ef.write(str(e))
            print(f"[ASR] {rec_id}: FAILED -> {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts_json", type=str, required=True,
                        help="Path to posts.json input file")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save ASR outputs")
    parser.add_argument("--device", choices=["cuda","cpu"], default=None)
    parser.add_argument("--compute_type", default=None)
    args = parser.parse_args()

    # Optional model overrides
    if args.device or args.compute_type:
        _ASR = WhisperModel(
            "medium",
            device=args.device if args.device else "cuda",
            compute_type=args.compute_type if args.compute_type else "float16"
        )

    process_posts(args.posts_json, args.out_dir)
