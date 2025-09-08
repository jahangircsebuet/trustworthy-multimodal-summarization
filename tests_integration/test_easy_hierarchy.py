from pathlib import Path

def test_hierarchy_local_and_fuse(tmp_path):
    text = (
        "Segment A: The conference opened with a keynote on trustworthy summarization. "
        "Attendees discussed retrieval augmentation and verification.\n\n"
        "Segment B: Later sessions focused on multimodal content like images and videos. "
        "A demo showed OCR and ASR feeding into a summary pipeline.\n\n"
        "Segment C: The closing panel emphasized evaluation, including QA-based factuality."
    )
    inp = tmp_path / "thread.txt"
    inp.write_text(text, encoding="utf-8")

    from src.segmentation.chunk import chunk_text
    chunks = chunk_text(text, max_tokens=60, overlap=10)
    assert len(chunks) >= 2

    # Preload a SMALL teacher for speed
    from src.teacher import local_summarizer as loc
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
    # Inject small model/tok just for this test run
    loc._tok, loc._model = tok, mdl

    summaries = [loc.summarize_chunk(c, max_new_tokens=64) for c in chunks]
    summaries = loc.dedupe_summaries(summaries, sim_thr=0.92)
    assert summaries and all(isinstance(s, str) for s in summaries)

    from src.teacher.fuse import fuse_summaries, _tok as fuse_tok, _model as fuse_model
    # Also preload small model for fuser
    from transformers import AutoTokenizer as T, AutoModelForSeq2SeqLM as M
    if fuse_tok is None or fuse_model is None:
        fuse_tok = T.from_pretrained("google/flan-t5-small")
        fuse_model = M.from_pretrained("google/flan-t5-small", device_map="auto")
    # Replace module globals
    import src.teacher.fuse as fuse_mod
    fuse_mod._tok, fuse_mod._model = fuse_tok, fuse_model

    global_sum = fuse_summaries(summaries, max_new_tokens=128)
    assert isinstance(global_sum, str) and len(global_sum.split()) > 10
