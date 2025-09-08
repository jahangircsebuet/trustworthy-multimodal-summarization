import types
import torch


def test_budget_decoder(monkeypatch, tmp_path):
    # Create a fake "model_dir" with tokenizer/model load behavior
    import src.decoding.budget as bd

    class FTok:
        def __call__(self, *a, **k): return {"input_ids": torch.tensor([[1]])}
        def decode(self, ids, skip_special_tokens=True): return "ok"
    class FModel:
        def __init__(self, *a, **k): self.device="cpu"
        def generate(self, **kw): return torch.tensor([[1,2,3]])

    monkeypatch.setattr(bd, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda p: FTok()))
    monkeypatch.setattr(bd, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=lambda p, **k: FModel()))

    dec = bd.BudgetDecoder("any")
    out, log = dec.generate_with_budget("text", budget_tokens=32, profile="quality")
    assert out == "ok" and log["budget_tokens"] == 32


def test_cost_tracker():
    from src.utils.cost import CostTracker
    with CostTracker() as ct:
        x = sum(range(10000))
    rep = ct.report()
    assert "wall_s" in rep and rep["wall_s"] >= 0.0
