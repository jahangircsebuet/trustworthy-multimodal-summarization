from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

import yaml

def load_yaml(path: str):
    """
    Load a YAML file and return a Python dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_special_tokens_if_missing(tokenizer) -> int:
    """
    Ensure tokenizer has PAD/BOS/EOS. Returns how many types were added.
    Also set right-padding (safer for causal LM SFT).
    """
    added = 0
    special_tokens = {}

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            special_tokens["pad_token"] = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            special_tokens["pad_token"] = tokenizer.sep_token
        else:
            special_tokens["pad_token"] = "<|pad|>"

    if tokenizer.eos_token is None and tokenizer.sep_token is None:
        special_tokens["eos_token"] = "</s>"

    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = "<s>"

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        added = len(special_tokens)

    tokenizer.padding_side = "right"
    return added


@dataclass
class DataCollatorForStrings:
    """
    Collator that takes dicts with 'source' and 'target' strings and returns
    {'input_ids','attention_mask','labels'} suitable for causal LM training.

    Strategy:
      input_ids = prompt_ids + target_ids(+eos)
      labels    = [-100]*len(prompt_ids) + target_ids(+eos)
      - If sequence too long, truncate from the LEFT of the prompt.
      - Then pad to batch max length.
    """
    tokenizer: Any
    max_length: int = 2048
    add_eos: bool = True
    prompt_template: Optional[str] = None  # e.g., "{source}\n\nTL;DR:\n"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            eos_id = self.tokenizer.eos_token_id

        for ex in features:
            src = ex.get("source", "")
            tgt = ex.get("target", "")

            prompt_text = self.prompt_template.format(source=src) if self.prompt_template else src

            prompt_ids = self.tokenizer(
                prompt_text, add_special_tokens=True, return_attention_mask=False, truncation=False
            )["input_ids"]

            target_ids = self.tokenizer(
                tgt, add_special_tokens=False, return_attention_mask=False, truncation=False
            )["input_ids"]

            if self.add_eos and (len(target_ids) == 0 or target_ids[-1] != eos_id):
                target_ids = target_ids + [eos_id]

            total_len = len(prompt_ids) + len(target_ids)
            if total_len > self.max_length:
                overflow = total_len - self.max_length
                if overflow >= len(prompt_ids):
                    prompt_ids = prompt_ids[-1:]  # keep at least one prompt token
                    new_total = len(prompt_ids) + len(target_ids)
                    if new_total > self.max_length:
                        trim = new_total - self.max_length
                        target_ids = target_ids[trim:]
                else:
                    prompt_ids = prompt_ids[overflow:]

            in_ids = prompt_ids + target_ids
            labs   = [-100] * len(prompt_ids) + target_ids

            input_ids_list.append(in_ids)
            labels_list.append(labs)

        batch_inputs = self.tokenizer.pad(
            {"input_ids": input_ids_list},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels_list},
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]

        pad_id = self.tokenizer.pad_token_id or eos_id
        batch_labels = batch_labels.masked_fill(batch_labels == pad_id, -100)

        return {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "labels": batch_labels,
        }
