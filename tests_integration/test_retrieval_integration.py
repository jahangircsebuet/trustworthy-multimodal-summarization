# tests/test_retrieval_integration.py
import importlib
from pathlib import Path
import sys
import numpy as np

# Ensure the project root (the dir that contains "src/") is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_web_retrieve_offline(monkeypatch):
    # Import your runtime-only retrieval module
    mod = importlib.import_module("src.retrieval.retrieve")

    # --- Stub external/network-dependent pieces so the test runs offline ---

    # 1) Fake Google results (two URLs)
    def fake_google_cse(query, num=10, lang=None, date_restrict=None):
        return [
            {"title": "Local News: Fireworks at Lakeside Park",
             "link": "https://news.example.com/lakeside", "snippet": "Fireworks show"},
            {"title": "Encyclopedia: Lakeside Park",
             "link": "https://en.wikipedia.org/wiki/Lakeside_Park", "snippet": "Park info"},
        ]
    monkeypatch.setattr(mod, "google_cse", fake_google_cse)

    # 2) Tiny deterministic embedding function (no real model download)
    def vec(text: str) -> np.ndarray:
        t = (text or "").lower()
        v = np.array([
            1.0 if "fireworks" in t else 0.0,
            1.0 if "lakeside"   in t else 0.0,
            1.0 if "chess"      in t else 0.0,
        ], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def fake_embed_texts(texts):
        arr = np.stack([vec(x) for x in texts], axis=0) if texts else np.zeros((0,1), dtype=np.float32)
        return arr
    # patch the name used inside src.retrieval.retrieve
    monkeypatch.setattr(mod, "embed_texts", fake_embed_texts)

    # 3) Stub page fetching / passage building
    def fake_build_web_passages(results, max_passages_per_page=5, passage_len=120):
        passages = [
            "Thousands attended a fireworks show at Lakeside Park on July 4, 2023.",
            "Lakeside Park is a public space with walking trails and a lake.",
        ]
        embs = np.stack([vec(p) for p in passages], axis=0)
        meta = [
            {"source": "web", "url": results[0]["link"], "title": results[0]["title"], "rank": 1},
            {"source": "web", "url": results[1]["link"], "title": results[1]["title"], "rank": 2},
        ]
        return embs, passages, meta
    monkeypatch.setattr(mod, "build_web_passages", fake_build_web_passages)

    # --- Call the function under test ---
    T_prime = ("On July 4, 2023, the city hosted a fireworks show at Lakeside Park. "
               "Families enjoyed music and food.")
    out = mod.web_retrieve(
        T_prime=T_prime,
        lang="en",
        k=2,
        queries=["lakeside fireworks 2023"],
        date_restrict=None,
        min_overlap=0.01,
        prefer_domains=["wikipedia.org"],
    )

    # --- Assertions ---
    assert isinstance(out, list) and len(out) == 2
    assert "meta" in out[0] and out[0]["meta"]["source"] == "web"
    assert "url" in out[0]["meta"]
    # The more on-topic passage (fireworks at Lakeside) should rank first
    assert "fireworks" in out[0]["text"].lower() and "lakeside" in out[0]["text"].lower()
    assert out[0]["score"] >= out[1]["score"]
