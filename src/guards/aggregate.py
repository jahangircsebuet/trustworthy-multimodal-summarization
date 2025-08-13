from typing import List, Dict

def decision(sent_sc: List[Dict], qa_thr=0.25, clip_thr=0.2):
    # sent_sc: [{"sent": str, "qa_score": float, "nli": "entailment|neutral|contradiction", "clip_max": float}]
    out = []
    for s in sent_sc:
        ok = (s["qa_score"] >= qa_thr) and (s["nli"] != "contradiction") and (s["clip_max"] >= clip_thr if s["clip_max"] is not None else True)
        out.append({"sent": s["sent"], "ok": ok, **s})
    return out
