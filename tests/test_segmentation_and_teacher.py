import types
import torch


def test_chunk_and_thread():
    from src.segmentation.chunk import chunk_text, build_thread_text
    txt = " ".join(["w"] * 2300)
    chunks = chunk_text(txt, max_tokens=512, overlap=64)
    assert len(chunks) >= 4
    thr = build_thread_text([{"timestamp":1,"text":"A"},{"timestamp":0,"text":"B"}])
    assert thr.strip().startswith("[0]")


def test_local_summarizer_and_dedupe(monkeypatch):
    import src.teacher.local_summarizer as loc
    # Patch summarizer model/tokenizer
    class FTok:
        def __call__(self, *a, **k): return {"input_ids": torch.tensor([[1]])}
        def decode(self, ids, skip_special_tokens=True): return "chunk summary"
        def to(self, device): return self
    class FModel:
        device="cpu"
        def generate(self, **kw): return torch.tensor([[1]])
    monkeypatch.setattr(loc, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda m: FTok()))
    monkeypatch.setattr(loc, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=lambda m, **k: FModel()))
    # Patch sentence-transformers & util
    class FEmb:
        def encode(self, xs, convert_to_tensor=True, normalize_embeddings=True):
            import torch
            return torch.eye(len(xs))
    class FUtil:
        @staticmethod
        def cos_sim(a, b):
            return (a @ b.T)  # identity -> identical strings = 1
    monkeypatch.setattr(loc, "SentenceTransformer", lambda *a, **k: FEmb())
    monkeypatch.setattr(loc, "util", FUtil)

    s = loc.summarize_chunk("..."*10)
    assert "chunk summary" in s
    deduped = loc.dedupe_summaries(["a","a","b"], sim_thr=0.5)
    assert deduped == ["a", "b"]
