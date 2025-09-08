import json
from pathlib import Path


def test_pipeline_smoke(monkeypatch, tmp_path, tiny_jsonl):
    """
    Smoke test the orchestration with stubs so we produce outputs without downloading models.
    """
    # Stubs
    monkeypatch.setattr("src.perception.ocr.run_ocr", lambda imgs: [])
    monkeypatch.setattr("src.perception.asr.run_asr", lambda vid: {})
    monkeypatch.setattr("src.perception.caption.caption_images", lambda imgs: [])
    monkeypatch.setattr("src.retrieval.retrieve.build_index_from_textbag",
                        lambda textbag_path, source="post": type("Idx", (), {"search": lambda self, q, topk=8: []})())
    monkeypatch.setattr("src.retrieval.retrieve.retrieve",
                        lambda q, indices, k=8: [{"text":"supporting text", "meta":{}} for _ in range(3)])
    monkeypatch.setattr("src.generation.generator.generate_summary", lambda prompt, **kw: "A draft summary.")
    monkeypatch.setattr("src.guards.qg.gen_questions", lambda s, max_q=2: ["What?"])
    monkeypatch.setattr("src.guards.qa.answer_with_context", lambda q, ctx: {"answer":"ok","score":0.9})
    monkeypatch.setattr("src.guards.nli.nli_label", lambda prem, hyp: "entailment")
    monkeypatch.setattr("src.guards.clip_align.clip_similarity", lambda p, t: 0.9)
    monkeypatch.setattr("src.guards.revise.generate_summary", lambda p, **kw: "A revised summary.")

    out_dir = tmp_path / "proc"
    from scripts.run_pipeline import main
    class Args: pass
    args = Args()
    args.input = tiny_jsonl
    args.out_dir = str(out_dir)
    main(args)

    # Check outputs
    draft = list(out_dir.glob("*_draft.txt"))
    revised = list(out_dir.glob("*_revised.txt"))
    assert draft and revised
