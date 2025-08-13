def build_prompt(evidence_items, task="Write a faithful 4-6 sentence summary grounded ONLY in the evidence. Add [refN] after each claim it uses."):
    ev_lines = []
    for i, e in enumerate(evidence_items, 1):
        ev_lines.append(f"[ref{i}] {e['text']}")
    ev_block = "\n".join(ev_lines)
    return f"""You are a careful summarizer.

EVIDENCE:
{ev_block}

TASK:
{task}
"""
