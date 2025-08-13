from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

_MODEL = None
_TOK = None

def load_gen(model_name="google/flan-t5-large"):
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return _TOK, _MODEL

def generate_summary(prompt: str, max_new_tokens=256, num_beams=4):
    tok, model = load_gen()
    inp = tok(prompt, return_tensors="pt", truncation=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tok.decode(out[0], skip_special_tokens=True)
