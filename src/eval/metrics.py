from rouge_score import rouge_scorer
from bert_score import score as bertscore
import numpy as np

def rouge_l(hyp: str, ref: str) -> float:
    sc = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    return sc.score(ref, hyp)['rougeLsum'].fmeasure

def bert_f1(hyp: str, ref: str) -> float:
    P, R, F1 = bertscore([hyp], [ref], lang="en", model_type="microsoft/deberta-xlarge-mnli")
    return float(F1.mean().item())

def aggregate(scores):
    return {k: float(np.mean([s[k] for s in scores])) for k in scores[0].keys()}
