# query_builder.py
import re
from typing import List

TRIGGER_WORDS = {"earthquake", "flood", "storm", "recall", "championship", "cup", "election", "summit"}

def build_queries_from_Tprime(T_prime: str, lang: str = "en") -> List[str]:
    """
    Simple heuristic: keep capitalized tokens (names), four-digit years, and trigger words.
    Replace later with spaCy/Stanza NER if you want.
    """
    caps = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", T_prime or "")
    years = re.findall(r"\b(19|20)\d{2}\b", T_prime or "")
    triggers = [w for w in TRIGGER_WORDS if re.search(rf"\b{re.escape(w)}\b", T_prime or "", re.I)]
    terms = [*caps, *years, *triggers]
    terms = [t.lower() for t in terms]
    # fallback if empty
    if not terms:
        terms = (T_prime or "").split()[:12]
    # One joined query; you can generate multiple if desired
    q = " ".join(dict.fromkeys(terms))  # dedupe, preserve order
    return [q]
