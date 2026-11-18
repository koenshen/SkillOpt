"""Tests for the QA split helpers used by the HotpotQA / TriviaQA materializers."""

import json

import pytest

from scripts.qa_split_utils import (
    check_splits,
    doc_block,
    join_doc_blocks,
    stratified_split,
    stratum_report,
    write_splits,
)


def _items(n: int, key_cycle=("a", "b", "c", "d")) -> list[dict]:
    return [
        {"id": f"q{i}", "question": f"q{i}", "context": "ctx", "answers": [f"a{i}"],
         "task_type": key_cycle[i % len(key_cycle)]}
        for i in range(n)
    ]


def key(item: dict) -> str:
    return item["task_type"]


def test_stratified_split_matches_requested_sizes() -> None:
    splits = stratified_split(_items(2800), {"train": 1600, "val": 200, "test": 1000}, key)
    assert {k: len(v) for k, v in splits.items()} == {"train": 1600, "val": 200, "test": 1000}


def test_stratified_split_keeps_stratum_shares_equal() -> None:
    splits = stratified_split(_items(2800), {"train": 1600, "val": 200, "test": 1000}, key)
    report = stratum_report(splits, key)
    for name in report:
        for share in report[name].values():
            assert abs(share - 0.25) < 0.02, report


def test_random_allocation_would_not_be_guaranteed_balanced() -> None:
    """Contiguous slicing is badly skewed; stratification is not."""
    items = [dict(item) for item in _items(2800)]
    # A contiguous slice over an ordered pool: 400 items are all type 'a'.
    for item in items[:400]:
        item["task_type"] = "a"
    splits = stratified_split(items, {"train": 1600, "val": 200, "test": 1000}, key)
    report = stratum_report(splits, key)
    shares = [report[name].get("a", 0.0) for name in splits]
    assert max(shares) - min(shares) < 0.02


def test_stratified_split_is_deterministic_for_the_same_seed() -> None:
    a = stratified_split(_items(2800), {"train": 1600, "val": 200, "test": 1000}, key, seed=7)
    b = stratified_split(_items(2800), {"train": 1600, "val": 200, "test": 1000}, key, seed=7)
    assert [i["id"] for i in a["train"]] == [i["id"] for i in b["train"]]
    c = stratified_split(_items(2800), {"train": 1600, "val": 200, "test": 1000}, key, seed=8)
    assert [i["id"] for i in c["train"]] != [i["id"] for i in a["train"]]


def test_stratified_split_never_reuses_an_item() -> None:
    splits = stratified_split(_items(2800), {"train": 1600, "val": 200, "test": 1000}, key)
    ids = [i["id"] for split in splits.values() for i in split]
    assert len(ids) == len(set(ids)) == 2800


def test_check_splits_rejects_imbalanced_composition() -> None:
    splits = {
        "train": _items(100, ("a",)),
        "val": _items(10, ("b",)),
        "test": _items(50, ("c",)),
    }
    with pytest.raises(SystemExit):
        check_splits(splits, key, tol=0.02)


def test_doc_block_and_join_respect_budget() -> None:
    blocks = [doc_block(f"T{i}", "x" * 1800) for i in range(5)]
    joined = join_doc_blocks(blocks, max_chars=6000)
    assert len(joined) <= 6000
    assert joined.startswith("[DOC] T0")
    assert joined.count("[DOC]") == 3


def test_doc_block_skips_empty_body() -> None:
    assert doc_block("T", "") == ""
    assert doc_block("", "body") == "[DOC]\nbody"


def test_write_splits_roundtrip(tmp_path) -> None:
    train = _items(3)
    val = [dict(item, id=f"v{i}") for i, item in enumerate(_items(2))]
    test = [dict(item, id=f"t{i}") for i, item in enumerate(_items(4))]
    splits = {"train": train, "val": val, "test": test}
    counts = write_splits(tmp_path, splits, source="unit-test", benchmark="demo")
    assert counts == {"train": 3, "val": 2, "test": 4}
    assert json.loads((tmp_path / "split_manifest.json").read_text())["benchmark"] == "demo"
    first = json.loads((tmp_path / "train" / "items.json").read_text())[0]
    assert set(first) >= {"id", "question", "context", "answers"}
