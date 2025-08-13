import io
import json
import os
from pathlib import Path

import PIL.Image as Image
import pytest


@pytest.fixture
def tmp_img(tmp_path):
    """Create a small RGB image and return its path."""
    p = tmp_path / "img.jpg"
    img = Image.new("RGB", (32, 32), color=(123, 222, 111))
    img.save(p)
    return str(p)


@pytest.fixture
def tmp_img2(tmp_path):
    p = tmp_path / "img2.jpg"
    img = Image.new("RGB", (32, 32), color=(111, 42, 222))
    img.save(p)
    return str(p)


@pytest.fixture
def tiny_jsonl(tmp_path):
    p = tmp_path / "dev.jsonl"
    items = [
        {"id": "p1", "text": "Hello world.", "images": [], "lang": "en"},
        {"id": "p2", "text": "Hola mundo.", "images": [], "lang": "es"},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return str(p)
