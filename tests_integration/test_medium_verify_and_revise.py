import re

def test_verify_and_revise_end2end(tmp_path):
    # Content with one supported and one dubious statement
    evidence_text = (
        "The city hosted a fireworks show at Lakeside Park on July 4, 2023, starting at 9 PM. "
        "Local bands performed before the show."
    )
    summary_draft = (
        "The city hosted a fireworks show at Lakeside Park on July 4, 2023 [ref1]. "
        "It began at 11 PM and featured international bands [ref1]."
    )

    # 1) Minimal evidence list (as if retrieved)
    evidence = [{"text": evidence_text}]

    # 2) Load real QG/QA/NLI models (smaller where possible)
    # Pre-load a small generator for later revision
    from src.generation.generator import load_gen
    load_gen("google/flan-t5-small")

    # (a) Question generation (base-sized but manageable)
    from src.guards.qg import gen_questions, load_qg
    load_qg("iarfmoose/t5-base-question-generator")  # one-time download
    qs = gen_questions(summary_draft, max_q=4)
    assert qs and all(q.endswith("?") for q in qs)

    # (b) QA against evidence (SQuAD2 base)
    from src.guards.qa import answer_with_context
    qa_scores = [answer_with_context(q, evidence_text)["score"] for q in qs]
    assert max(qa_scores) >= 0.1  # some confidence should show up

    # (c) NLI with a lighter checkpoint (distilbert MNLI)
    from src.guards.nli import load_nli, nli_label
    load_nli("typeform/distilbert-base-uncased-mnli")
    nli = nli_label(evidence_text, "It began at 11 PM.")
    assert nli in {"entailment", "neutral", "contradiction"}

    # 3) Build per-sentence scores and aggregate
    from nltk import download as nltk_dl
    nltk_dl("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(summary_draft)
    sent_sc = []
    for s in sents:
        # simple QA score for the sentence: best of generated qs
        best = 0.0
        for q in gen_questions(s, max_q=2):
            best = max(best, answer_with_context(q, evidence_text)["score"])
        lbl = nli_label(evidence_text, s)
        sent_sc.append({"sent": s, "qa_score": best, "nli": lbl, "clip_max": None})

    from src.guards.aggregate import decision
    flags = decision(sent_sc, qa_thr=0.25, clip_thr=0.2)
    assert any(not f["ok"] for f in flags)  # at least one sentence should be flagged

    # 4) Ask the generator to revise given issues + evidence
    from src.guards.revise import revise_summary
    revised = revise_summary(summary_draft, flags, evidence)
    assert isinstance(revised, str) and len(revised) > 0
    # Heuristic: the time "11 PM" should be gone or hedged
    assert "11 PM" not in revised
