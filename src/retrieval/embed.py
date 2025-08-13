from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np, re

_EMB = None

def get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    global _EMB
    if _EMB is None:
        _EMB = SentenceTransformer(model_name, device="cuda" if hasattr(__import__("torch").cuda, "is_available") and __import__("torch").cuda.is_available() else "cpu")
    return _EMB

def split_into_passages(text: str, max_len=256) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    passages, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= max_len:
            passages.append(" ".join(cur)); cur = []
    if cur: passages.append(" ".join(cur))
    return passages

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    return np.asarray(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True))
