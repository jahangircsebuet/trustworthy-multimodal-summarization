import re
import torch

from typing import List
from transformers import pipeline

_TRANSLATOR = None

def get_translator(model_name: str = "Helsinki-NLP/opus-mt-mul-en"):
    """
    Lazily create a MarianMT translation pipeline.
    Uses CUDA if available; otherwise CPU. Doesn't require accelerate.
    """
    global _TRANSLATOR
    if _TRANSLATOR is None:
        
        device = 0 if torch.cuda.is_available() else -1
        _TRANSLATOR = pipeline(
            task="translation",
            model=model_name,
            device=device,         # use explicit device, not device_map
            # use_fast=True is fine; Marian still needs sentencepiece installed
        )
    return _TRANSLATOR


def _sent_split(text: str) -> List[str]:
    # crude sentence splitter; swap for spaCy if you prefer
    return re.split(r'(?<=[.!?])\s+', text.strip())

def translate_long(text: str, max_chunk_chars: int = 800) -> str:
    chunks, buf = [], []
    cur_len = 0
    for s in _sent_split(text):
        if cur_len + len(s) > max_chunk_chars and buf:
            chunks.append(" ".join(buf)); buf, cur_len = [], 0
        buf.append(s); cur_len += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))

    translator = get_translator()
    outs = translator(chunks, max_length=512)
    return " ".join(o["translation_text"] for o in outs)

def to_english(texts: List[str]) -> List[str]:
    return [translate_long(t) if len(t) > 900 else get_translator()(t, max_length=512)[0]["translation_text"]
            for t in texts]

if __name__ == "__main__":
    # Example usage
    texts = ["Bonjour le monde", "Hola mundo", "Hallo Welt"]
    translations = to_english(texts)
    for original, translated in zip(texts, translations):
        print(f"{original} -> {translated}")
    # Output:
    # Bonjour le monde -> Hello world
    # Hola mundo -> Hello world
    # Hallo Welt -> Hello world
    # (actual translations may vary based on the model and input)
    # Ensure you have the necessary model downloaded; this may take time on first run.
    # You can adjust the model to a specific language pair if needed.
    # For example, use "Helsinki-NLP/opus-mt-fr-en" for French to English.
    # Make sure to have the transformers library installed and configured properly.
    # pip install transformers
    # You may need to set up a Hugging Face account and authenticate if the model requires it.
