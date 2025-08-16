"""
Fuse module for combining multiple summaries.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global variables for the test to inject
_tok = None
_model = None

def fuse_summaries(summaries, max_new_tokens=128):
    """
    Combine multiple summaries into a single coherent summary.
    """
    global _tok, _model
    
    if _tok is None or _model is None:
        raise Exception("Model not loaded. Call load_model() first.")
    
    if not summaries:
        return ""
    
    # Combine all summaries into a single prompt
    combined_text = " ".join(summaries)
    
    # Create a prompt for fusion
    prompt = f"Combine these summaries into one coherent summary: {combined_text}"
    
    # Generate fused summary
    inputs = _tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            early_stopping=True
        )
    
    fused_summary = _tok.decode(outputs[0], skip_special_tokens=True)
    return fused_summary


def load_model(model_name="google/flan-t5-small"):
    """
    Load the model and tokenizer.
    """
    global _tok, _model
    _tok = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    return _tok, _model 