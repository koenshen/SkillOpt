"""Regression tests for the final==best reuse paths in the trainer.

When the final skill is identical to the best skill the trainer skips a
redundant rollout. The test set branch must still leave a per-item
``results.jsonl`` behind, and the selection set branch must leave a marker,
so downstream paired analysis never sees a half-empty directory.
"""

import json
from dataclasses import dataclass, field

from skillopt.engine.trainer import _write_results_jsonl


def test_write_results_jsonl_accepts_plain_dicts(tmp_path) -> None:
    out = tmp_path / "test_eval_final"
    _write_results_jsonl(
        [
            {"id": "a", "hard": 1, "soft": 1.0},
            {"id": "b", "hard": 0, "soft": 0.0, "fail_reason": "EM mismatch"},
        ],
        str(out),
    )
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[1]["fail_reason"] == "EM mismatch"


@dataclass
class _FakeRolloutResult:
    id: str
    hard: float
    soft: float
    fail_reason: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "hard": self.hard, "soft": self.soft, "fail_reason": self.fail_reason}


def test_write_results_jsonl_prefers_to_dict(tmp_path) -> None:
    out = tmp_path / "final_selection_eval"
    _write_results_jsonl([_FakeRolloutResult(id="x", hard=1, soft=0.9)], str(out))
    row = json.loads((out / "results.jsonl").read_text().strip())
    assert row == {"id": "x", "hard": 1, "soft": 0.9, "fail_reason": ""}


def test_write_results_jsonl_falls_back_to_dunder_dict(tmp_path) -> None:
    out = tmp_path / "d"


    class Obj:
        def __init__(self) -> None:
            self.id = "y"
            self.hard = 1

    _write_results_jsonl([Obj()], str(out))
    row = json.loads((out / "results.jsonl").read_text().strip())
    assert row["id"] == "y"


def test_write_results_jsonl_creates_directory_and_empty_file(tmp_path) -> None:
    out = tmp_path / "nested" / "test_eval_final"
    _write_results_jsonl([], str(out))
    assert (out / "results.jsonl").exists()
    assert (out / "results.jsonl").read_text() == ""
