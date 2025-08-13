from transformers import pipeline
from typing import List

# Generic multi→en translator; adjust languages as needed.
_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device_map="auto")

def to_english(texts: List[str]) -> List[str]:
    outs = _translator(texts, max_length=512)
    return [o["translation_text"] for o in outs]
