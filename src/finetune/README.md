# Trustworthy Summarization – Fine-Tuning Setup

This repository contains the fine-tuning setup for social network summarization aligned with our paper draft.  

---

## ✅ Why this matches our plan

- **Social-only baseline**  
  Matches platform style (slang, brevity, casual tone).  

- **Mixed domain with domain tags + sampling ratios**  
  Boosts coherence and factuality without losing social tone.  

- **LoRA/QLoRA**  
  Fast, budget-friendly fine-tuning on 8–24GB GPUs.  

- **Evaluation hooks**  
  ROUGE + BERTScore baked in for automatic quality checks.  

---

## 📂 Structure

- `configs/` – YAML configs for social-only and mixed-domain runs  
- `data_mix/` – dataset loaders, formatters, and sampling ratio mixer  
- `trainer/` – training loop, eval metrics, and utilities  
- `run_social.py` – entry point for fine-tuning on social datasets only  
- `run_mixed.py` – entry point for fine-tuning on social + news datasets  

---

## 🚀 Quickstart

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
