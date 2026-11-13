"""Tests for the BIRD text-to-SQL environment and its split scripts."""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import make_bird_id_split as split_mod
from scripts import materialize_bird as mat_mod
from skillopt.envs.bird import rollout as rollout_mod
from skillopt.envs.bird.evaluator import execution_match, extract_sql


# ── SQL extraction (same behaviour as the reference run_prompt.py) ──────────


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Reasoning...\n```sql\nSELECT a FROM t;\n```", "SELECT a FROM t;"),
        ("```sqlite\nSELECT a FROM t WHERE b = 'x;y';\nSELECT 2;\n```", "SELECT a FROM t WHERE b = 'x;y';"),
        ("think\nSELECT name FROM t\nORDER BY 1", "SELECT name FROM t\nORDER BY 1"),
        ("```\nWITH c AS (SELECT 1) SELECT * FROM c\n```", "WITH c AS (SELECT 1) SELECT * FROM c"),
        ("SELECT 'it''s;' AS v; trailing", "SELECT 'it''s;' AS v;"),
        ("", ""),
    ],
)
def test_extract_sql(response: str, expected: str) -> None:
    assert extract_sql(response) == expected


# ── Execution accuracy ───────────────────────────────────────────────────────


@pytest.fixture()
def db_root(tmp_path: Path) -> Path:
    root = tmp_path / "dbs"
    (root / "shop").mkdir(parents=True)
    conn = sqlite3.connect(root / "shop" / "shop.sqlite")
    conn.executescript(
        "CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT, price REAL);"
        "INSERT INTO item VALUES (1, 'pen', 1.5), (2, 'cup', 3.0), (3, 'pen', 2.0);"
    )
    conn.commit()
    conn.close()
    return root


def _db(db_root: Path) -> str:
    return str(db_root / "shop" / "shop.sqlite")


def test_set_comparison_ignores_order_and_duplicates(db_root: Path) -> None:
    gold = "SELECT name FROM item ORDER BY id"
    assert execution_match("SELECT DISTINCT name FROM item ORDER BY name DESC", gold, _db(db_root))["res"] == 1


def test_extra_column_is_wrong(db_root: Path) -> None:
    out = execution_match("SELECT name, price FROM item", "SELECT name FROM item", _db(db_root))
    assert out["res"] == 0 and out["status"] == "ok"


def test_execution_error_scores_zero(db_root: Path) -> None:
    out = execution_match("SELECT nope FROM item", "SELECT name FROM item", _db(db_root))
    assert out["res"] == 0 and out["status"] == "error"


def test_write_statements_are_blocked(db_root: Path) -> None:
    out = execution_match("DELETE FROM item", "SELECT name FROM item", _db(db_root))
    assert out["res"] == 0 and out["status"] == "error"
    conn = sqlite3.connect(_db(db_root))
    assert conn.execute("SELECT COUNT(*) FROM item").fetchone()[0] == 3
    conn.close()


def test_timeout_scores_zero(db_root: Path) -> None:
    slow = (
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
        "SELECT COUNT(*) FROM n"
    )
    out = execution_match(slow, "SELECT 1", _db(db_root), timeout=0.5)
    assert out["res"] == 0 and out["status"] == "timeout"


# ── Rollout ──────────────────────────────────────────────────────────────────


def _item(item_id: str, gold: str) -> dict:
    return {
        "id": item_id,
        "question_id": int(item_id),
        "db_id": "shop",
        "difficulty": "simple",
        "question": "Which items?",
        "evidence": "",
        "prompt": "Schema...\nQuestion: Which items?",
        "SQL": gold,
    }


def test_run_batch_scores_and_resumes(monkeypatch: pytest.MonkeyPatch, db_root: Path, tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_chat_target(system, user, **kwargs):
        calls.append({"system": system, "user": user})
        return "```sql\nSELECT name FROM item;\n```", {}

    monkeypatch.setattr(rollout_mod, "chat_target", fake_chat_target)
    monkeypatch.setattr(rollout_mod, "is_target_exec_backend", lambda: False)
    items = [_item("1", "SELECT name FROM item"), _item("2", "SELECT price FROM item")]
    out_dir = tmp_path / "out"

    first = rollout_mod.run_batch(items[:1], str(out_dir), "# Skill", str(db_root), workers=2)
    assert [r["hard"] for r in first] == [1]
    assert calls[0]["system"] == "# Skill" and calls[0]["user"] == items[0]["prompt"]

    results = rollout_mod.run_batch(items, str(out_dir), "# Skill", str(db_root), workers=2)
    assert len(calls) == 2  # item 1 was resumed, not re-run
    by_id = {r["id"]: r for r in results}
    assert by_id["1"]["hard"] == 1 and by_id["2"]["hard"] == 0
    assert by_id["2"]["fail_reason"].startswith("EX=0: result mismatch")
    conv = json.loads((out_dir / "predictions" / "2" / "conversation.json").read_text())
    assert "Gold SQL: SELECT price FROM item" in conv[-1]["content"]


# ── Split scripts ────────────────────────────────────────────────────────────


def _rows(n: int, start: int = 0) -> list[dict]:
    dbs, diffs = ["a", "b", "c"], ["simple", "moderate", "challenging"]
    return [
        {"question_id": i, "db_id": dbs[i % 3], "difficulty": diffs[(i // 3) % 3]}
        for i in range(start, start + n)
    ]


def test_build_split_is_deterministic_and_disjoint() -> None:
    dev = _rows(300)
    mini = [r for r in dev if r["question_id"] % 5 == 0]
    a = split_mod.build_split(dev, mini, val_size=60, seed=42)
    b = split_mod.build_split(dev, mini, val_size=60, seed=42)
    assert a == b
    ids = {k: {r["question_id"] for r in v} for k, v in a.items()}
    assert len(ids["val"]) == 60 and len(ids["test"]) == 60 and len(ids["train"]) == 180
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])


def test_build_split_rejects_mini_not_in_dev() -> None:
    with pytest.raises(ValueError, match="not in dev"):
        split_mod.build_split(_rows(10), _rows(2, start=100), val_size=2, seed=42)


def test_materialize_uses_mini_rows_for_test(tmp_path: Path) -> None:
    def full(r: dict, sql: str) -> dict:
        return {**r, "question": "q", "evidence": "e", "SQL": sql, "prompt": "p"}

    dev_rows = _rows(30)
    mini_rows = [r for r in dev_rows if r["question_id"] < 6]
    split = split_mod.build_split(dev_rows, mini_rows, val_size=6, seed=1)
    manifest = tmp_path / "ids"
    split_mod.write_split(split, manifest, Namespace(dev_jsonl=Path("d.jsonl"), mini_dev_jsonl=Path("m.jsonl"), seed=1))

    dev = {str(r["question_id"]): full(r, "SELECT dev") for r in dev_rows}
    mini = {str(r["question_id"]): full(r, "SELECT mini") for r in mini_rows}
    counts = mat_mod.materialize(manifest, tmp_path / "out", dev, mini)
    assert counts == {"train": 18, "val": 6, "test": 6}
    test_items = json.loads((tmp_path / "out" / "test" / "items.json").read_text())
    assert {it["SQL"] for it in test_items} == {"SELECT mini"}
    assert all(it["task_type"] == it["difficulty"] for it in test_items)
