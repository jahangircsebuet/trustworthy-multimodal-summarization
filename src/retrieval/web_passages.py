# web_passages.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import trafilatura
from .embed import split_into_passages, embed_texts

def fetch_main_text(url: str, timeout: int = 20) -> str:
    downloaded = trafilatura.fetch_url(url, timeout=timeout)
    if not downloaded:
        return ""
    txt = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_recall=False,
    )
    return txt or ""

def build_web_passages(results: List[Dict], max_passages_per_page: int = 5,
                       passage_len: int = 120) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    From google_cse() results -> fetch page -> extract article -> split to passages -> embed all passages.
    Returns (embs, passages, meta), with embs normalized for cosine/IP.
    fetch, extract, chunk, embed
    """
    passages: List[str] = []
    meta: List[Dict] = []
    for rank, r in enumerate(results, 1):
        url = r["link"]
        text = fetch_main_text(url)
        if not text:
            continue
        parts = split_into_passages(text, max_len=passage_len)[:max_passages_per_page]
        for p in parts:
            passages.append(p)
            meta.append({"source": "web", "url": url, "title": r.get("title", ""), "rank": rank})
    if not passages:
        return np.zeros((0,1), dtype=np.float32), [], []
    embs = embed_texts(passages)  # normalized
    return embs, passages, meta
