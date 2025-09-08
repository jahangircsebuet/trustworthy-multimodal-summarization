# train_sft.py
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .utils import DataCollatorForStrings, add_special_tokens_if_missing
from .eval_metrics import compute_metrics

def load_model_and_tokenizer(model_name: str, gradient_checkpointing: bool = True, bf16: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
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

def train(
    dataset_dict,
    cfg,
):
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

    args = TrainingArguments(
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
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        report_to="none",
        max_steps=cfg.get("max_steps", -1),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda p: compute_metrics(
            preds=[s.strip() for s in tokenizer.batch_decode(p.predictions, skip_special_tokens=True)],
            refs=[s.strip() for s in tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)],
        ),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    if hasattr(model, "peft_config"):  # LoRA
        model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
