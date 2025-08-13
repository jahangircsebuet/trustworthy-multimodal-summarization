from typing import List
from ..generation.generator import generate_summary
from ..generation.prompts import build_prompt

def revise_summary(draft: str, flags: List[dict], evidence_items: List[dict]):
    issues = []
    for f in flags:
        if not f["ok"]:
            reason = []
            if f["qa_score"] < 0.25: reason.append("unsupported by evidence")
            if f["nli"] == "contradiction": reason.append("contradicted by evidence")
            if f.get("clip_max") is not None and f["clip_max"] < 0.2: reason.append("misaligned with image")
            issues.append(f"- \"{f['sent']}\" is " + " & ".join(reason))
    if not issues:  # nothing to fix
        return draft
    instructions = "Revise the summary to remove or hedge the problematic claims below. Keep only statements supported by evidence and retain citations.\n" + "\n".join(issues)
    prompt = build_prompt(evidence_items, task=instructions + "\n\nOriginal summary:\n" + draft + "\n\nRevised, faithful summary:")
    return generate_summary(prompt, max_new_tokens=256, num_beams=4)
