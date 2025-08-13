import json
from typing import Iterator, List, Dict

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def iter_batches(items: List, bs: int = 8) -> Iterator[List]:
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

def save_jsonl(recs: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
