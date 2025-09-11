import json
import matplotlib.pyplot as plt
import os

# Base directory to save figures
base_url = "/home/malam10/projects/trustworthy-multimodal-summarization/src/visualizations"
os.makedirs(base_url, exist_ok=True)

# Paths to trainer_state.json files
paths = {
    "checkpoint-100": "/home/malam10/projects/trustworthy-multimodal-summarization/src/finetune/outputs/social_only_large/checkpoint-100/trainer_state.json",
    "checkpoint-200": "/home/malam10/projects/trustworthy-multimodal-summarization/src/finetune/outputs/social_only_large/checkpoint-200/trainer_state.json",
    "checkpoint-219": "/home/malam10/projects/trustworthy-multimodal-summarization/src/finetune/outputs/social_only_large/checkpoint-219/trainer_state.json",
}

# Steps to extract
target_steps = {50, 100, 150, 200}

# Collect metrics
results = {step: {"train_loss": None, "eval_loss": None, "rouge1": None,
                  "rouge2": None, "rougeL": None, "bertscore": None}
           for step in target_steps}

for ckpt, path in paths.items():
    with open(path) as f:
        data = json.load(f)
    for log in data["log_history"]:
        step = log.get("step")
        if step in target_steps:
            if "loss" in log and not log.get("eval_loss"):  # training loss
                results[step]["train_loss"] = log["loss"]
            if "eval_loss" in log:
                results[step]["eval_loss"] = log["eval_loss"]
                results[step]["rouge1"] = log.get("eval_rouge1")
                results[step]["rouge2"] = log.get("eval_rouge2")
                results[step]["rougeL"] = log.get("eval_rougeL")
                results[step]["bertscore"] = log.get("eval_bertscore_f1")

# Sort by step
steps = sorted(results.keys())
x = range(len(steps))

# --- Plot 1: Training vs Eval Loss ---
train_losses = [results[s]["train_loss"] for s in steps]
eval_losses = [results[s]["eval_loss"] for s in steps]

width = 0.35
plt.figure(figsize=(7,5))
plt.bar([i-width/2 for i in x], train_losses, width, label="Training Loss")
plt.bar([i+width/2 for i in x], eval_losses, width, label="Eval Loss")
plt.xticks(x, steps)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{base_url}/loss_comparison.png", dpi=300)
plt.close()

# --- Plot 2: ROUGE Scores ---
rouge1 = [results[s]["rouge1"] for s in steps]
rouge2 = [results[s]["rouge2"] for s in steps]
rougeL = [results[s]["rougeL"] for s in steps]

plt.figure(figsize=(7,5))
plt.bar([i-0.2 for i in x], rouge1, width=0.2, label="ROUGE-1")
plt.bar(x, rouge2, width=0.2, label="ROUGE-2")
plt.bar([i+0.2 for i in x], rougeL, width=0.2, label="ROUGE-L")
plt.xticks(x, steps)
plt.xlabel("Training Steps")
plt.ylabel("ROUGE Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{base_url}/rouge_scores.png", dpi=300)
plt.close()

# --- Plot 3: BERTScore F1 ---
bertscores = [results[s]["bertscore"] for s in steps]

plt.figure(figsize=(7,5))
plt.bar(x, bertscores, width=0.5, color="purple")
plt.xticks(x, steps)
plt.xlabel("Training Steps")
plt.ylabel("BERTScore F1")
plt.tight_layout()
plt.savefig(f"{base_url}/bertscore.png", dpi=300)
plt.close()

print("✅ Saved figures in:", base_url)
