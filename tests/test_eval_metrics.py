import types


def test_metrics_monkeypatched(monkeypatch):
    import src.eval.metrics as m

    # Patch BERTScore to avoid downloads
    def fake_score(hyps, refs, lang="en", model_type=None):
        import numpy as np
        return (None, None, [0.7])  # P, R, F1

    monkeypatch.setattr(m, "bertscore", types.SimpleNamespace(score=fake_score))

    r = m.rouge_l("a b c", "a b c")
    b = m.bert_f1("hello", "hello")
    assert r > 0.5 and b == 0.7
