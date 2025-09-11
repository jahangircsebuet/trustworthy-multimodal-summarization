import os
import torch
import warnings
from typing import Dict, Any, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    __version__ as transformers_version,
)
from packaging import version
from peft import LoraConfig, get_peft_model
from .utils import DataCollatorForStrings, add_special_tokens_if_missing
from .eval_metrics import compute_metrics

# Load environment variables from .env file
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")
warnings.filterwarnings("ignore", message=".*No label_names provided.*")
warnings.filterwarnings("ignore", message=".*max_length.*ignored.*")


def load_model_and_tokenizer(model_name: str, gradient_checkpointing: bool = True, bf16: bool = True):
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print("Warning: HUGGINGFACE_HUB_TOKEN not found in .env file")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        token=hf_token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bool(bf16 and torch.cuda.is_available())
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=hf_token,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=hf_token,
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        try:
            model.config.use_cache = False
        except Exception:
            pass

    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

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
    Construct TrainingArguments with proper evaluation settings.
    Handles both new (eval_strategy) and old (evaluation_strategy) Transformers versions.
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
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        report_to=["none"],
        max_steps=cfg.get("max_steps", -1),
        load_best_model_at_end=cfg.get("load_best_model_at_end", False),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        remove_unused_columns=False,
        save_strategy="steps",
        logging_strategy="steps",
    )

    if version.parse(transformers_version) >= version.parse("4.30.0"):
        return TrainingArguments(eval_strategy="steps", **base_kwargs)
    else:
        return TrainingArguments(evaluation_strategy="steps", **base_kwargs)


def _decode_predictions_and_refs(predictions, label_ids, tokenizer) -> Tuple[list, list]:
    pred_ids = predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.detach().cpu().numpy()
    import numpy as np
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

    collator = DataCollatorForStrings(
        tokenizer=tokenizer,
        max_length=cfg["max_source_len"],
    )

    args = _build_training_args(cfg)

    def _metrics_fn(p):
        preds_text, refs_text = _decode_predictions_and_refs(p.predictions, p.label_ids, tokenizer)
        return compute_metrics(preds=preds_text, refs=refs_text)

    print("Dataset structure:")
    print(f"Train dataset columns: {dataset_dict['train'].column_names}")
    print(f"Train dataset size: {len(dataset_dict['train'])}")
    print(f"Validation dataset size: {len(dataset_dict['validation'])}")
    
    train_dataset = dataset_dict["train"]
    if "input_ids_text" in train_dataset.column_names and "labels_text" in train_dataset.column_names:
        print("Renaming columns to match collator expectations...")
        train_dataset = train_dataset.rename_columns({
            "input_ids_text": "source",
            "labels_text": "target"
        })
        val_dataset = dataset_dict["validation"].rename_columns({
            "input_ids_text": "source", 
            "labels_text": "target"
        })
        print(f"Train dataset columns after renaming: {train_dataset.column_names}")
    else:
        val_dataset = dataset_dict["validation"]

    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=_metrics_fn,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=_metrics_fn,
        )

    print("Training configuration:")
    print(f"  - Total training steps: {len(train_dataset) // (cfg['per_device_train_batch_size'] * cfg['gradient_accumulation_steps']) * cfg['num_train_epochs']}")
    print(f"  - Evaluation every: {cfg['eval_steps']} steps")
    print(f"  - Logging every: {cfg['logging_steps']} steps")

    trainer.train()
    
    print("Running final evaluation...")
    final_metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {final_metrics}")
    
    import csv
    csv_path = os.path.join(cfg["output_dir"], "final_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in final_metrics.items():
            try:
                v = float(v)
            except Exception:
                v = str(v)
            w.writerow([k, v])
    print(f"Saved final metrics to {csv_path}")

    trainer.save_model(cfg["output_dir"])
    if hasattr(model, "peft_config"):
        model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
