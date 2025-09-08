# formatters.py
from typing import Dict

SYSTEM_PROMPT = (
    "You are a helpful assistant that writes concise, faithful summaries. "
    "Prefer short, direct sentences. Do not add facts not present in the input."
)

def apply_domain_tags(example: Dict, domain: str, use_tags: bool) -> Dict:
    """
    Prepend control tokens for domain and optional length.
    """
    if use_tags:
        example["source"] = f"<domain={domain}>\n" + example["source"]
    return example

def format_chat(example: Dict, max_target_len: int = 128) -> Dict:
    """
    Convert into instruction-style single-turn format.
    """
    prompt = (
        f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n[/SYSTEM]\n"
        f"[USER]\nSummarize the following content in {max_target_len} tokens or fewer.\n"
        f"{example['source']}\n[/USER]\n[ASSISTANT]\n"
    )
    return {
        "input_ids_text": prompt,       # we tokenize later
        "labels_text": example["target"]
    }
