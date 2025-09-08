# websearch.py
from __future__ import annotations
import os, httpx
from typing import List, Dict, Optional

GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"

def google_cse(query: str, num: int = 10, lang: Optional[str] = None,
               date_restrict: Optional[str] = None) -> List[Dict]:
    """
    Query Google Programmable Search (Custom Search JSON API).
    - lang: e.g., 'en', used as lr=lang_en to bias language.
    - date_restrict: 'd7' (7 days), 'w2' (2 weeks), 'm3' (3 months), etc.
    Returns: [{title, link, snippet}, ...]
    Google Programmable Search API
    """
    key = os.environ["GOOGLE_API_KEY"]
    cx  = os.environ["GOOGLE_CSE_ID"]
    params = {"q": query, "key": key, "cx": cx, "num": min(max(num,1), 10)}
    if lang and lang != "auto":
        params["lr"] = f"lang_{lang}"
    if date_restrict:
        params["dateRestrict"] = date_restrict

    r = httpx.get(GOOGLE_CSE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for it in data.get("items", []):
        out.append({"title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")})
    return out
