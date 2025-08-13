import numpy as np
from .embed import embed_texts, get_embedder
from .index import FaissIndex

def build_index_from_textbag(textbag_path: str, source="post") -> FaissIndex:
    with open(textbag_path, "r", encoding="utf-8") as f:
        txt = f.read()
    from .embed import split_into_passages
    passages = split_into_passages(txt, max_len=120)
    embs = embed_texts(passages)
    idx = FaissIndex(embs.shape[1])
    idx.add(embs, passages, [{"source": source, "ref": i} for i in range(len(passages))])
    return idx

def retrieve(query: str, indices, k=8):
    emb = get_embedder().encode([query], normalize_embeddings=True, convert_to_numpy=True)
    cands = []
    for idx in indices:
        cands += idx.search(emb, topk=k)
    cands = sorted(cands, key=lambda x: x["score"], reverse=True)
    # Deduplicate by text
    seen, out = set(), []
    for c in cands:
        if c["text"] in seen: continue
        seen.add(c["text"]); out.append(c)
        if len(out) >= k: break
    return out
