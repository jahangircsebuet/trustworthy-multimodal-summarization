from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

_MODEL = None
_TOK = None

def load_gen(model_name="google/flan-t5-large"):
    global _MODEL, _TOK
    if _MODEL is None:
        try:
            # Try to load without authentication first
            _TOK = AutoTokenizer.from_pretrained(model_name, token=None, trust_remote_code=True, local_files_only=False)
            _MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=None, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=False)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            # Try with local_files_only=True to use only cached models
            try:
                print("Trying to load from local cache only...")
                _TOK = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                _MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
            except Exception as e2:
                print(f"Failed to load from local cache: {e2}")
                # Try a different approach - use a simple model or mock
                raise Exception(f"Could not load any model. First error: {e}, Second error: {e2}")
    return _TOK, _MODEL

def generate_summary(prompt: str, max_new_tokens=256, num_beams=4):
    tok, model = load_gen()
    inp = tok(prompt, return_tensors="pt", truncation=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tok.decode(out[0], skip_special_tokens=True)
