# dataset_loaders.py
import json
from typing import Iterable, Dict, Any, Optional
from datasets import load_dataset, Dataset, DatasetDict

def load_reddit_tifu(subset: str = "short", split: str = "train") -> Dataset:
    """
    HF: reddit_tifu with subset 'short' or 'long'
    Fields:
      - document: post text
      - tldr: summary
    """
    ds = load_dataset("Fredithefish/Reddit-TIFU", subset, split=split)
    return ds.remove_columns([c for c in ds.column_names if c not in {"document","tldr"}])

def load_tweetsumm_local(jsonl_path: str) -> Dataset:
    """
    Local JSONL with fields:
      - thread: list[str] or concatenated string of tweets
      - summary: str
    If thread is list, we join with ' <TSEP> ' markers.
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            thread = ex.get("thread")
            if isinstance(thread, list):
                source = " <TSEP> ".join(thread)
            else:
                source = str(thread)
            rows.append({"source": source, "target": ex.get("summary","")})
    return Dataset.from_list(rows)

def load_cnn_dm(version: str = "3.0.0", split: str = "train") -> Dataset:
    """
    HF: cnn_dailymail
    Fields:
      - article -> source
      - highlights -> target
    """
    ds = load_dataset("cnn_dailymail", version, split=split)
    return ds.rename_columns({"article":"source","highlights":"target"}).remove_columns(
        [c for c in ds.column_names if c not in {"source","target"}]
    )

def load_multi_news(split: str = "train") -> Dataset:
    """
    HF: multi_news
    Fields:
      - document -> source
      - summary -> target
    """
    ds = load_dataset("multi_news", split=split)
    return ds.rename_columns({"document":"source","summary":"target"}).remove_columns(
        [c for c in ds.column_names if c not in {"source","target"}]
    )

def to_unified(ds, source_field: str, target_field: str) -> Dataset:
    return ds.rename_columns({source_field: "source", target_field: "target"}).remove_columns(
        [c for c in ds.column_names if c not in {"source","target"}]
    )

def load_social_datasets(cfg) -> Iterable[Dataset]:
    if cfg["datasets"]["reddit_tifu"]["use"]:
        ds = load_reddit_tifu(cfg["datasets"]["reddit_tifu"]["subset"], cfg["datasets"]["reddit_tifu"]["split"])
        yield to_unified(ds, "document", "tldr")
    if cfg["datasets"].get("tweetsumm", {}).get("use", False):
        yield load_tweetsumm_local(cfg["datasets"]["tweetsumm"]["path"])

def load_news_datasets(cfg) -> Iterable[Dataset]:
    if cfg["datasets"].get("cnn_dailymail", {}).get("use", False):
        p = cfg["datasets"]["cnn_dailymail"]
        yield load_cnn_dm(p["version"], p["split"])
    if cfg["datasets"].get("multi_news", {}).get("use", False):
        yield load_multi_news(cfg["datasets"]["multi_news"]["split"])
