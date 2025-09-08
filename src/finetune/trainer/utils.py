# utils.py
import yaml, os, torch
from dataclasses import dataclass
from typing import Dict, List
from transformers import AutoTokenizer

@dataclass
class DataCollatorForStrings:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, batch):
        inputs = [ex["input_ids_text"] for ex in batch]
        labels = [ex["labels_text"] for ex in batch]
        model_inputs = self.tokenizer(
            inputs,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels_tok = self.tokenizer(
                labels, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
            )
        # Replace pad tokens by -100 for loss masking
        labels_ids = labels_tok["input_ids"]
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels_ids
        return model_inputs

def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def add_special_tokens_if_missing(tokenizer):
    added = 0
    specials = ["<domain=social>", "<domain=news>", "<TSEP>"]
    for tok in specials:
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens([tok])
            added += 1
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        added += 1
    return added
