import types
import torch


def test_fuse_summaries(monkeypatch):
    import src.teacher.fuse as fu

    class FTok:
        def __call__(self, *a, **k): return {"input_ids": torch.tensor([[1]])}
        def decode(self, ids, skip_special_tokens=True): return "global fused summary"
        def to(self, device): return self
    class FModel:
        device="cpu"
        def generate(self, **kw): return torch.tensor([[1]])

    monkeypatch.setattr(fu, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda m: FTok()))
    monkeypatch.setattr(fu, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=lambda m, **k: FModel()))
    out = fu.fuse_summaries(["s1","s2"])
    assert "global" in out


def test_build_pairs_dataset(tmp_path):
    from src.distill.make_dataset import build_pairs
    from datasets import DatasetDict

    chunks = ["c1", "c2"]
    locals_ = ["s1", "s2"]
    ds = build_pairs(chunks, locals_, "s1\ns2", "global")
    assert isinstance(ds, DatasetDict)
    assert "train" in ds and len(ds["train"]) >= 2
