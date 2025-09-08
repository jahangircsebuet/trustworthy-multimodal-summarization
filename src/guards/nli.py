from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

_NLI_TOK = None
_NLI = None
LABELS = ["entailment", "neutral", "contradiction"]

def load_nli(model="microsoft/deberta-large-mnli"):
    global _NLI_TOK, _NLI
    if _NLI is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Always use cuda:0
        _NLI_TOK = AutoTokenizer.from_pretrained(model)
        _NLI = AutoModelForSequenceClassification.from_pretrained(model).to(device)
    return _NLI_TOK, _NLI

def nli_label(premise: str, hypothesis: str) -> str:
    tok, m = load_nli()
    device = next(m.parameters()).device
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = m(**inputs).logits[0].softmax(-1)
    return LABELS[int(torch.argmax(logits))]