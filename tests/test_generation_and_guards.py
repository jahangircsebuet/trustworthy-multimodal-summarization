import types
import torch
import pytest


def test_build_prompt():
    from src.generation.prompts import build_prompt
    ev = [{"text": "A"}, {"text": "B"}]
    p = build_prompt(ev, task="Summarize X.")
    assert "EVIDENCE:" in p and "Summarize X." in p and "[ref1]" in p


def test_generate_summary_monkeypatched(monkeypatch):
    import src.generation.generator as gen

    class FakeTok:
        def __call__(self, text, return_tensors="pt", truncation=True): return {"input_ids": torch.tensor([[1,2]])}
        def decode(self, ids, skip_special_tokens=True): return "OK"
        def to(self, device): return self

    class FakeModel:
        device = "cpu"
        def generate(self, **kw): return torch.tensor([[1,2,3]])

    monkeypatch.setattr(gen, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda m: FakeTok()))
    monkeypatch.setattr(gen, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=lambda m, **k: FakeModel()))
    out = gen.generate_summary("prompt", max_new_tokens=10, num_beams=2)
    assert out == "OK"


def test_qg_gen_questions(monkeypatch):
    import src.guards.qg as qg

    class FakeTok:
        def __call__(self, text, return_tensors="pt"): return {"input_ids": torch.tensor([[1]])}
        def decode(self, ids, skip_special_tokens=True): return "When? Where?"

    class FakeModel:
        def generate(self, **kw): return torch.tensor([[1]])

    monkeypatch.setattr(qg, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda m: FakeTok()))
    monkeypatch.setattr(qg, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=lambda m, device_map="auto": FakeModel()))
    qs = qg.gen_questions("A happened in B.")
    assert qs == ["When?", "Where?"]


def test_qa_answer(monkeypatch):
    import src.guards.qa as qa

    class FakePipe:
        def __call__(self, question, context, handle_impossible_answer=True, top_k=1):
            return {"answer": "42", "score": 0.9}

    monkeypatch.setattr(qa, "pipeline", lambda *a, **k: FakePipe())
    qa._QA = FakePipe()
    out = qa.answer_with_context("q", "c")
    assert out["answer"] == "42" and out["score"] > 0.5


def test_nli_label(monkeypatch):
    import src.guards.nli as nli

    class FakeTok:
        def __call__(self, a, b, return_tensors="pt", truncation=True): return {"input_ids": torch.tensor([[1,2]])}

    class FakeModel:
        def __init__(self): self.device = "cpu"
        def __call__(self, **kw):
            class O: logits = torch.tensor([[0.1, 0.2, 0.7]])
            return O()

    monkeypatch.setattr(nli, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda m: FakeTok()))
    monkeypatch.setattr(nli, "AutoModelForSequenceClassification",
                        types.SimpleNamespace(from_pretrained=lambda m, device_map="auto": FakeModel()))
    lab = nli.nli_label("prem", "hypo")
    assert lab == "contradiction"


def test_clip_similarity(monkeypatch, tmp_img):
    import src.guards.clip_align as ca

    class FakeModel:
        def __init__(self): self.logit_scale = types.SimpleNamespace(device="cpu")
        def encode_image(self, img): return torch.tensor([[1.0, 0.0]])
        def encode_text(self, txt): return torch.tensor([[1.0, 0.0]])

    def fake_create(*a, **k): return FakeModel(), (lambda im: im), None
    def fake_tok(name): return lambda xs: torch.tensor([[1,2]])

    monkeypatch.setattr(ca, "open_clip", types.SimpleNamespace(create_model_and_transforms=fake_create, get_tokenizer=fake_tok))
    sim = ca.clip_similarity(tmp_img, "anything")
    assert sim == pytest.approx(1.0)


def test_decision_and_revise(monkeypatch):
    from src.guards.aggregate import decision
    from src.guards.revise import revise_summary

    flags = decision([
        {"sent":"A", "qa_score":0.3, "nli":"entailment", "clip_max":0.5},
        {"sent":"B", "qa_score":0.1, "nli":"contradiction", "clip_max":0.1},
    ])
    assert flags[0]["ok"] is True and flags[1]["ok"] is False

    # monkeypatch generator to return revised text
    import src.guards.revise as rv
    monkeypatch.setattr(rv, "generate_summary", lambda *a, **k: "REVISED")
    out = revise_summary("draft", flags, [{"text":"ev1"}])
    assert out in ("draft", "REVISED")
