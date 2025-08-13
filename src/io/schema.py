from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class PostRecord:
    id: str
    text: str
    images: List[str] = None
    video: Optional[str] = None
    lang: str = "en"
    meta: Dict = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "PostRecord":
        return PostRecord(
            id=d["id"],
            text=d.get("text", ""),
            images=d.get("images", []) or [],
            video=d.get("video"),
            lang=d.get("lang", "en"),
            meta=d.get("meta", {}) or {}
        )
