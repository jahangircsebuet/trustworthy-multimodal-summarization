# src/finetune/trainer/eval_metrics.py
from typing import List, Dict
from rouge_score import rouge_scorer, scoring
from bert_score import score as bertscore_score

# Configure ROUGE
_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
_R_SCORER = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)

def compute_metrics(preds: List[str], refs: List[str], lang: str = "en") -> Dict:
    # Aggregate ROUGE
    aggregator = scoring.BootstrapAggregator()
    for p, r in zip(preds, refs):
        aggregator.add_scores(_R_SCORER.score(r, p))  # (reference, prediction)
    r_agg = aggregator.aggregate()
    out = {
        "rouge1": r_agg["rouge1"].mid.fmeasure,
        "rouge2": r_agg["rouge2"].mid.fmeasure,
        "rougeL": r_agg["rougeL"].mid.fmeasure,
    }
    # BERTScore (returns P, R, F1 tensors)
    P, R, F1 = bertscore_score(preds, refs, lang=lang, rescale_with_baseline=True)
    out["bertscore_f1"] = float(F1.mean().item()) if len(F1) > 0 else 0.0
    return out
