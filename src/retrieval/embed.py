# embed.py
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np, re

_EMB = None

def get_embedder(model_name=None):
    """Multilingual by default; inner product with normalized vectors == cosine."""
    global _EMB
    if model_name is None:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if _EMB is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMB = SentenceTransformer(model_name, device=device)
    return _EMB

def split_into_passages(text: str, max_len=120) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    passages, cur = [], []
    for s in sents:
        if not s:
            continue
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= max_len:
            passages.append(" ".join(cur)); cur = []
    if cur: passages.append(" ".join(cur))
    return passages

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = get_embedder()
    return np.asarray(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True))
