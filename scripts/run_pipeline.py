import argparse, os, nltk, re
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from src.io.load import load_jsonl
from src.perception.ocr import run_ocr
from src.perception.asr import run_asr
from src.perception.caption import caption_images
from src.perception.pack_text import pack_text
from src.retrieval.retrieve import build_index_from_textbag, retrieve
from src.generation.prompts import build_prompt
from src.generation.generator import generate_summary
from src.guards.qg import gen_questions
from src.guards.qa import answer_with_context
from src.guards.nli import nli_label
from src.guards.clip_align import clip_similarity
from src.guards.aggregate import decision
from src.guards.revise import revise_summary

def main(args):
    recs = load_jsonl(args.input)
    os.makedirs(args.out_dir, exist_ok=True)
    for rec in recs:
        rid = rec["id"]
        images = rec.get("images", []) or []
        video = rec.get("video")
        ocr = run_ocr(images) if images else []
        asr = run_asr(video) if video else {}
        caps = caption_images(images) if images else []
        textbag_path = pack_text(args.out_dir, rid, rec.get("text",""), ocr, asr, caps)

        post_idx = build_index_from_textbag(textbag_path, source="post")
        # (Optional) add KB index here and pass [post_idx, kb_idx]; we keep it simple:
        evidence = retrieve("what happened who where when why", [post_idx], k=8)
        prompt = build_prompt(evidence)
        draft = generate_summary(prompt)

        # Verification
        sents = [s for s in sent_tokenize(draft) if len(s.split()) > 3]
        sent_scores = []
        context = "\n".join(e["text"] for e in evidence)
        for s in sents:
            qas = gen_questions(s, max_q=2)
            qa_best = 0.0
            for q in qas:
                a = answer_with_context(q, context)
                qa_best = max(qa_best, a["score"])
            nli = nli_label(context, s)
            clip_max = None
            if images:
                clip_max = max(clip_similarity(p, s) for p in images)
            sent_scores.append({"sent": s, "qa_score": qa_best, "nli": nli, "clip_max": clip_max})

        flags = decision(sent_scores)
        revised = revise_summary(draft, flags, evidence)

        with open(os.path.join(args.out_dir, f"{rid}_draft.txt"), "w", encoding="utf-8") as f: f.write(draft)
        with open(os.path.join(args.out_dir, f"{rid}_revised.txt"), "w", encoding="utf-8") as f: f.write(revised)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with records")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()
    main(args)
