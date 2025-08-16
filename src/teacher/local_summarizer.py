"""
Local summarizer module for testing purposes.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global variables for the test to inject
_tok = None
_model = None

def summarize_chunk(chunk, max_new_tokens=64):
    """
    Summarize a text chunk using the loaded model.
    """
    global _tok, _model
    
    if _tok is None or _model is None:
        raise Exception("Model not loaded. Call load_model() first.")
    
    # Simple summarization logic for testing
    inputs = _tok(chunk, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            early_stopping=True
        )
    
    summary = _tok.decode(outputs[0], skip_special_tokens=True)
    return summary


def dedupe_summaries(summaries, sim_thr=0.92):
    """
    Remove duplicate summaries based on similarity threshold.
    Simple implementation for testing.
    """
    if not summaries:
        return []
    
    # Simple deduplication - remove exact duplicates
    unique_summaries = []
    for summary in summaries:
        if summary not in unique_summaries:
            unique_summaries.append(summary)
    
    return unique_summaries


def load_model(model_name="google/flan-t5-small"):
    """
    Load the model and tokenizer.
    """
    global _tok, _model
    _tok = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    return _tok, _model 