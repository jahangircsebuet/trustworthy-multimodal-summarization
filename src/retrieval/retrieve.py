# retrieve.py
from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
from .embed import embed_texts
from .websearch import google_cse
from .web_passages import build_web_passages
from .filters import keyword_overlap, prefer_domain

# runtime-only, Google-backed retrieval

def web_retrieve(
    T_prime: str,
    lang: str = "en",
    k: int = 8,
    queries: Optional[List[str]] = None,
    web_pages_per_query: int = 8,
    date_restrict: Optional[str] = None,      # e.g., 'm6'
    min_overlap: float = 0.03,
    prefer_domains: Optional[List[str]] = None,  # e.g., ['wikipedia.org','reuters.com','bbc.com']
    block_domains: Optional[List[str]] = None
) -> List[Dict]:
    """
    Runtime-only retrieval:
      1) Build or accept queries from T'
      2) Google search each query
      3) Fetch pages, extract main text, chunk to passages, embed
      4) Rank passages by cosine similarity to T' embedding
      5) Apply lexical relevance + domain preference
      6) Return top-k [{score, text, meta}]
    """
    prefer_set = set(prefer_domains or [])
    block_set   = set(block_domains or [])

    # 1) Embed T' as the query representation
    q_emb = embed_texts([T_prime])  # shape (1, dim)

    # 2) Queries
    if queries is None:
        from .query_builder import build_queries_from_Tprime
        queries = build_queries_from_Tprime(T_prime, lang)

    # 3) Aggregate Google results across queries
    serps: List[Dict] = []
    seen_links = set()
    for q in queries:
        res = google_cse(q, num=min(web_pages_per_query, 10), lang=lang, date_restrict=date_restrict)
        for r in res:
            if r["link"] in seen_links:
                continue
            seen_links.add(r["link"])
            serps.append(r)

    # 4) Fetch -> passages -> embeddings
    embs, passages, meta = build_web_passages(serps, max_passages_per_page=5, passage_len=120)

    if not passages:
        return []

    # 5) Cosine similarity (since normalized)
    scores = (embs @ q_emb.T).reshape(-1)  # (N,)

    # 6) Post-filter & rank
    cands = []
    for i, (s, p, m) in enumerate(zip(scores, passages, meta)):
        if keyword_overlap(queries[0], p) < min_overlap:
            continue
        dom_bonus = prefer_domain(m["url"], prefer=prefer_set, block=block_set)
        # small domain prior: -1, 0, +1 -> map to [ -0.02, 0, +0.02 ]
        final_score = float(s) + 0.02 * dom_bonus
        cands.append({"score": final_score, "text": p, "meta": m})

    # 7) Sort, dedupe by text, keep top-k
    cands.sort(key=lambda x: x["score"], reverse=True)
    seen_text, out = set(), []
    for c in cands:
        t = c["text"]
        if t in seen_text:
            continue
        seen_text.add(t)
        out.append(c)
        if len(out) >= k:
            break

    return out


if __name__ == "__main__":
    D_items = web_retrieve(
        T_prime,
        lang=post.lang or "en",
        k=8,
        date_restrict="m6",  # prefer last 6 months
        prefer_domains=["wikipedia.org","reuters.com","bbc.com","apnews.com","nature.com"]
    )
    D = [x["text"] for x in D_items]  # snippets for your generator
    for d in D:
        print(d)