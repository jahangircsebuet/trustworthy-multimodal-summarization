import os
import torch
from typing import Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from .utils import DataCollatorForStrings, add_special_tokens_if_missing
from .eval_metrics import compute_metrics


def load_model_and_tokenizer(model_name: str, gradient_checkpointing: bool = True, bf16: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    use_bf16 = bool(bf16 and torch.cuda.is_available())
    # Prefer dtype on newer TF; older still accept torch_dtype (may warn but OK)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # avoid HF warning & ensure compatibility
        try:
            model.config.use_cache = False
        except Exception:
            pass

    added = add_special_tokens_if_missing(tokenizer)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def maybe_wrap_lora(model, lora_cfg: dict):
    if not lora_cfg.get("enabled", False):
        return model
    lc = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lc)
    return model


def _build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    """
    Construct TrainingArguments, auto-selecting the correct strategy key:
      - Try eval_strategy (Transformers v5)
      - Fallback to evaluation_strategy (Transformers v4)
    """
    base_kwargs: Dict[str, Any] = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        bf16=cfg.get("bf16", True),
        logging_steps=cfg["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        report_to=["none"],
        max_steps=cfg.get("max_steps", -1),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        remove_unused_columns=False,  # let our custom collator handle columns
    )

    # Allow explicit overrides if present
    if "logging_strategy" in cfg:
        base_kwargs["logging_strategy"] = cfg["logging_strategy"]
    if "save_strategy" in cfg:
        base_kwargs["save_strategy"] = cfg["save_strategy"]

    strat_val = cfg.get("evaluation_strategy", cfg.get("eval_strategy", "steps"))
    try:
        return TrainingArguments(eval_strategy=strat_val, **base_kwargs)  # v5
    except TypeError:
        return TrainingArguments(evaluation_strategy=strat_val, **base_kwargs)  # v4


def _decode_predictions_and_refs(predictions, label_ids, tokenizer) -> Tuple[list, list]:
    """
    Robust decoding:
      - If predictions are logits, argmax -> ids
      - If tuple, take first item
      - Replace -100 in labels with pad/eos id before decoding
    """
    pred_ids = predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.detach().cpu().numpy()
    import numpy as np  # noqa
    if getattr(pred_ids, "ndim", 0) == 3:
        pred_ids = pred_ids.argmax(-1)

    labels = label_ids
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels = labels.copy()
    labels[labels == -100] = pad_id

    preds_text = [s.strip() for s in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    refs_text  = [s.strip() for s in tokenizer.batch_decode(labels,   skip_special_tokens=True)]
    return preds_text, refs_text


def train(dataset_dict, cfg):
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        bf16=cfg.get("bf16", True),
    )
    model = maybe_wrap_lora(model, cfg["lora"])

    # Build collator
    collator = DataCollatorForStrings(
        tokenizer=tokenizer,
        max_length=cfg["max_source_len"],
    )

    args = _build_training_args(cfg)

    def _metrics_fn(p):
        preds_text, refs_text = _decode_predictions_and_refs(p.predictions, p.label_ids, tokenizer)
        return compute_metrics(preds=preds_text, refs=refs_text)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,        # OK on v4; v5 deprecates but still works
        data_collator=collator,
        compute_metrics=_metrics_fn,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    if hasattr(model, "peft_config"):  # LoRA
        model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
