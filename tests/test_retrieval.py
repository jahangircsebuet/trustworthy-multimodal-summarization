import numpy as np
import types
import pytest


def test_split_and_embed(monkeypatch):
    import src.retrieval.embed as emb

    # fake embedder returns deterministic vectors
    class FakeST:
        def __init__(self, name, device="cpu"): pass
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            # deterministic by length
            return np.array([[len(t) % 7 + 1.0] * 4 for t in texts], dtype=float)

    monkeypatch.setattr(emb, "SentenceTransformer", FakeST)
    m = emb.get_embedder()
    vecs = emb.embed_texts(["a", "bb", "ccc"])
    assert vecs.shape == (3, 4)
    parts = emb.split_into_passages("a. b? c! d. e.")
    assert len(parts) >= 2


def test_faiss_index_roundtrip(monkeypatch, tmp_path):
    faiss = pytest.importorskip("faiss")
    from src.retrieval.index import FaissIndex
    idx = FaissIndex(dim=4)
    embs = np.eye(3, 4, dtype=np.float32)
    idx.add(embs, ["p0", "p1", "p2"], [{"source":"s","ref":i} for i in range(3)])

    q = np.array([[1,0,0,0]], dtype=np.float32)
    res = idx.search(q, topk=2)
    assert len(res) == 2 and res[0]["text"] == "p0"

    idx.save(str(tmp_path))
    idx2 = FaissIndex.load(str(tmp_path))
    res2 = idx2.search(q, topk=1)
    assert res2[0]["text"] == "p0"


def test_build_index_and_retrieve(monkeypatch, tmp_path):
    import src.retrieval.retrieve as rt
    import src.retrieval.embed as emb

    # Prepare a textbag
    textbag = tmp_path / "tb.txt"
    textbag.write_text("Alpha bravo. Charlie delta echo. Foxtrot golf hotel.")
    # Fake embedder that makes second sentence most similar to 'charlie'
    class FakeST:
        def __init__(self, name, device="cpu"): pass
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            # Query is ["charlie"], passages are 3; make middle one highest
            if isinstance(texts, list) and len(texts)==1:
                return np.array([[1,2,3,4]], dtype=float)
            return np.array([
                [0.1, 0.2, 0.3, 0.4],
                [9.9, 9.9, 9.9, 9.9],  # highest score passage
                [0.2, 0.3, 0.4, 0.5],
            ], dtype=float)

    monkeypatch.setattr(emb, "SentenceTransformer", FakeST)
    idx = rt.build_index_from_textbag(str(textbag))
    res = rt.retrieve("charlie", [idx], k=1)
    assert "Charlie" in res[0]["text"]
