from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

_NLI_TOK = None
_NLI = None
LABELS = ["entailment", "neutral", "contradiction"]

def load_nli(model="microsoft/deberta-v3-large-mnli"):
    global _NLI_TOK, _NLI
    if _NLI is None:
        _NLI_TOK = AutoTokenizer.from_pretrained(model)
        _NLI = AutoModelForSequenceClassification.from_pretrained(model, device_map="auto")
    return _NLI_TOK, _NLI

def nli_label(premise: str, hypothesis: str) -> str:
    tok, m = load_nli()
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True).to(m.device)
    with torch.no_grad():
        logits = m(**inputs).logits[0].softmax(-1)
    return LABELS[int(torch.argmax(logits))]
