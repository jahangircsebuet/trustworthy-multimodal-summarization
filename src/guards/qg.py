from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, re

_QG_TOK = None
_QG = None

def load_qg(model_name="iarfmoose/t5-base-question-generator"):
    global _QG_TOK, _QG
    if _QG is None:
        _QG_TOK = AutoTokenizer.from_pretrained(model_name)
        _QG = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    return _QG_TOK, _QG

def gen_questions(summary: str, max_q=6):
    tok, m = load_qg()
    prompt = f"generate questions: {summary}"
    ids = tok(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ids, max_new_tokens=128, num_beams=4)
    text = tok.decode(out[0], skip_special_tokens=True)
    # naive split by "?" and cleanup
    qs = [q.strip()+"?" for q in re.split(r'\?\s*', text) if q.strip()]
    return qs[:max_q]
