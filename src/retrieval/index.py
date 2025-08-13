import faiss, numpy as np, os, json
from typing import List, Dict

class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.passages: List[str] = []
        self.meta: List[Dict] = []

    def add(self, embs: np.ndarray, passages: List[str], meta: List[Dict]):
        assert embs.shape[0] == len(passages) == len(meta)
        self.index.add(embs.astype(np.float32))
        self.passages += passages
        self.meta += meta

    def search(self, query_emb: np.ndarray, topk=8):
        D, I = self.index.search(query_emb.astype(np.float32), topk)
        out = []
        for d, i in zip(D[0], I[0]):
            if i == -1: continue
            out.append({"score": float(d), "text": self.passages[i], "meta": self.meta[i]})
        return out

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "store.json"), "w", encoding="utf-8") as f:
            json.dump({"passages": self.passages, "meta": self.meta}, f)

    @staticmethod
    def load(path: str):
        idx = FaissIndex(1)  # placeholder, will overwrite
        idx.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "store.json"), "r", encoding="utf-8") as f:
            js = json.load(f)
        idx.passages = js["passages"]; idx.meta = js["meta"]
        return idx
