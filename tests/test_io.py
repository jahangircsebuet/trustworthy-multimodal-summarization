import json
from src.io.schema import PostRecord
from src.io.load import load_jsonl, save_jsonl, iter_batches


def test_postrecord_roundtrip():
    pr = PostRecord(id="x", text="t", images=["a"], video=None, lang="en", meta={"k": 1})
    d = pr.to_dict()
    pr2 = PostRecord.from_dict(d)
    assert pr == pr2
    assert pr2.images == ["a"]


def test_load_save_jsonl(tmp_path):
    recs = [{"id": "a", "text": "t1"}, {"id": "b", "text": "t2"}]
    p = tmp_path / "a.jsonl"
    save_jsonl(recs, str(p))
    got = load_jsonl(str(p))
    assert got == recs


def test_iter_batches():
    xs = list(range(10))
    batches = list(iter_batches(xs, bs=3))
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]
