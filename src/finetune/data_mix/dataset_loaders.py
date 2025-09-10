import json
from typing import Iterable, Optional, List
from datasets import load_dataset, Dataset

def _keep_only(ds: Dataset, allowed: set) -> Dataset:
    """Drop all columns except those in `allowed`."""
    drop = [c for c in ds.column_names if c not in allowed]
    return ds.remove_columns(drop) if drop else ds

def _rename_then_drop(ds: Dataset, mapping: dict) -> Dataset:
    """
    Rename columns using `mapping`, then drop everything except {'source','target'}.
    This two-step avoids the 'column not found' error.
    """
    ds = ds.rename_columns(mapping)
    drop = [c for c in ds.column_names if c not in {"source", "target"}]
    return ds.remove_columns(drop) if drop else ds

# -----------------------------
# Social sources
# -----------------------------
def load_reddit_tifu(split: str = "train") -> Dataset:
    """
    Normalize Reddit-TIFU to {source, target}.
    Fredithefish/Reddit-TIFU exposes: text, title, (meta).
    If a fork exposes document/tldr, handle that too.
    """
    ds = load_dataset("Fredithefish/Reddit-TIFU", "default", split=split)
    cols = set(ds.column_names)

    if {"text", "title"}.issubset(cols):
        ds = _keep_only(ds, {"text", "title"})
        return _rename_then_drop(ds, {"text": "source", "title": "target"})

    # there is no document, tldr in dataset 
    # if {"document", "tldr"}.issubset(cols):
    #     ds = _keep_only(ds, {"document", "tldr"})
    #     return _rename_then_drop(ds, {"document": "source", "tldr": "target"})

    raise ValueError(
        f"Unsupported Reddit-TIFU schema: {sorted(cols)}. "
        "Expected either {text,title} or {document,tldr}."
    )

def load_tweetsumm_local(jsonl_path: str) -> Dataset:
    """
    Load a local JSONL file with 'thread' and 'summary'.
    If thread is a list, join with ' <TSEP> '. Output {source, target}.
    """
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            thread = ex.get("thread")
            source = " <TSEP> ".join(thread) if isinstance(thread, list) else str(thread)
            rows.append({"source": source, "target": ex.get("summary", "")})
    return Dataset.from_list(rows)

# -----------------------------
# News sources
# -----------------------------
def load_cnn_dm(version: str = "3.0.0", split: str = "train") -> Dataset:
    """
    Load cnn_dailymail and normalize to {source, target}.
    Maps article → source, highlights → target, then drops extras.
    """
    ds = load_dataset("cnn_dailymail", version, split=split)
    return _rename_then_drop(ds, {"article": "source", "highlights": "target"})

def _normalize_multi_news_schema(ds: Dataset) -> Optional[Dataset]:
    """
    Try several common Multi-News schemas and normalize to {source, target}.
    Returns normalized dataset or None if unrecognized.
    """
    cols = set(ds.column_names)

    # Classic: document/summary
    if {"document", "summary"}.issubset(cols):
        return _rename_then_drop(ds, {"document": "source", "summary": "target"})

    # Already normalized
    if {"source", "target"}.issubset(cols):
        drop = [c for c in ds.column_names if c not in {"source", "target"}]
        return ds.remove_columns(drop) if drop else ds

    # GEM-style: source + references (list[str])
    if {"source", "references"}.issubset(cols):
        def map_first_ref(example):
            refs = example.get("references") or []
            return {"target": refs[0] if isinstance(refs, list) and refs else ""}
        ds2 = ds.map(map_first_ref, remove_columns=[c for c in ds.column_names if c != "source"])
        drop = [c for c in ds2.column_names if c not in {"source", "target"}]
        return ds2.remove_columns(drop) if drop else ds2

    return None

def load_multi_news(split: str = "train") -> Dataset:
    """
    Load Multi-News robustly under datasets>=3.0.
    Tries mirrors that ship data (not loader scripts) first; normalizes schema.
    """
    candidates: List[str] = [
        "GEM/multi_news",          # non-script mirror (preferred)
        "alexfabbri/multi_news",   # original (likely script-based; may fail on v3)
    ]

    last_err = None
    for repo in candidates:
        try:
            ds = load_dataset(repo, split=split)
        except Exception as e:
            last_err = e
            continue

        norm = _normalize_multi_news_schema(ds)
        if norm is not None:
            return norm

    hint = (
        "All Multi-News mirrors failed under datasets>=3.0 or had unknown schema.\n"
        "Options:\n"
        "  - Set datasets.multi_news.use=false (skip Multi-News)\n"
        "  - Or pin: pip install 'datasets==2.19.*'  # re-enable legacy script datasets\n"
        "  - Or provide a repo with parquet/json columns (document/summary or source/target)"
    )
    if last_err:
        raise RuntimeError(hint) from last_err
    raise RuntimeError(hint)

# -----------------------------
# Generic utility (optional)
# -----------------------------
def to_unified(ds: Dataset, source_field: str, target_field: str) -> Dataset:
    """
    Rename any dataset with fields {source_field, target_field} to {source, target},
    then drop everything else.
    """
    if source_field not in ds.column_names or target_field not in ds.column_names:
        raise ValueError(
            f"to_unified expected {{'{source_field}','{target_field}'}}, "
            f"but dataset has {ds.column_names}"
        )
    return _rename_then_drop(ds, {source_field: "source", target_field: "target"})

# -----------------------------
# Entry points used by run scripts
# -----------------------------
def load_social_datasets(cfg) -> Iterable[Dataset]:
    if cfg["datasets"]["reddit_tifu"]["use"]:
        yield load_reddit_tifu(cfg["datasets"]["reddit_tifu"]["split"])
    if cfg["datasets"].get("tweetsumm", {}).get("use", False):
        yield load_tweetsumm_local(cfg["datasets"]["tweetsumm"]["path"])

def load_news_datasets(cfg) -> Iterable[Dataset]:
    if cfg["datasets"].get("cnn_dailymail", {}).get("use", False):
        p = cfg["datasets"]["cnn_dailymail"]
        yield load_cnn_dm(p["version"], p["split"])

    if cfg["datasets"].get("multi_news", {}).get("use", False):
        # Let training continue if Multi-News fails
        allow_skip = cfg["datasets"]["multi_news"].get("allow_skip", True)
        try:
            yield load_multi_news(cfg["datasets"]["multi_news"]["split"])
        except Exception as e:
            if allow_skip:
                print("[warn] Skipping multi_news:", e)
            else:
                raise
