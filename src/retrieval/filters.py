# filters.py
import re
from urllib.parse import urlparse
# light lexical gate + domain preferences

def _tokset(s: str):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

def keyword_overlap(q: str, passage: str) -> float:
    qs, ps = _tokset(q), _tokset(passage)
    if not qs or not ps:
        return 0.0
    return len(qs & ps) / max(1, len(qs | ps))

def prefer_domain(url: str, prefer: set[str] | None = None, block: set[str] | None = None) -> int:
    """
    Return a small bonus/penalty bucket by domain. Higher is better.
    e.g., prefer={"wikipedia.org","reuters.com"}, block={"random-blog.example"}
    """
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return 0
    if block and any(host.endswith(b) for b in block):
        return -1
    if prefer and any(host.endswith(p) for p in prefer):
        return 1
    return 0
