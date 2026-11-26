import hashlib
import json

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skillopt_evolution.archive import load_archive


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sheet(path, text, points):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "results.jsonl").open("w") as f:
        for key, hard in points.items():
            f.write(json.dumps({"id": key, "hard": hard, "agent_ok": True}) + "\n")
    for key in points:
        prompt = path / "predictions" / key / "target_system_prompt.txt"
        prompt.parent.mkdir(parents=True)
        # BIRD stores only the stripped skill in its system message.
        prompt.write_text(text.strip())


def make_archive(tmp_path):
    write(tmp_path / "config.json", {"env": "bird"})
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "skill_v0000.md").write_text("parent\n")
    sheet(tmp_path / "selection_eval_baseline", "parent", {"c": 1, "b": 0, "a": 0, "d": 1})
    s = tmp_path / "steps" / "step_0001"
    sheet(s / "selection_eval", "new", {"b": 0, "a": 0})
    sheet(s / "rollout", "parent", {"train": 1})
    (s / "candidate_skill.md").write_text("new")
    write(s / "ranked_edits.json", {"edits": [{"op": "append", "content": "new"}]})
    write(s / "edit_apply_report.json", [{"index": 1, "op": "append", "status": "applied_append"}])
    (skills / "skill_v0001.md").write_text("parent\n")
    (skills / "skill_v0002.md").write_text("parent\n")
    (tmp_path / "best_skill.md").write_text("parent\n")
    write(tmp_path / "history.json", [
        {"step": 1, "action": "reject", "tokens": {"rollout": {"prompt_tokens": 12, "completion_tokens": 2}},
         "candidate_hash": hashlib.sha256(b"new").hexdigest()[:16], "selection_hard": 0.5,
         "selection_early_stop": {"n_evaluated": 2, "n_total": 4, "batch_size": 2, "stopped": True,
                                  "batches": [{"n_evaluated": 2, "hard_sum": 0}]}},
        {"step": 2, "action": "skip_no_patches"},
    ])
    write(tmp_path / "final_selection_eval" / "selection.json", {"reused_from": "best"})
    return tmp_path


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.archive = make_archive(Path(temp.name))

    def test_real_prefix_partial_parent_and_final_reuse(self):
        archive = self.archive
        data = load_archive(archive, list("abcd"))
        assert [r["id"] for r in data["population"]] == ["v0"]
        assert data["initial_population"]["shortfall"] == 4
        parent, child = data["records"]
        assert child["correct"] is None and child["score"] is None
        assert child["parent_sha256"] == parent["sha256"]
        assert child["edits"] == [dict(index=1, op="append", target="", content="new", status="applied_append")]
        assert data["history"] == {"a": [0, 0], "b": [0, 0], "c": [1], "d": [1]}
        assert data["native_step_usage"]["input_tokens"] == 12
        assert [s["id"] for s in data["skipped"]] == ["step_0002", "final"]


    def test_real_final_uses_last_retained_text_and_full_score(self):
        archive = self.archive
        final = archive / "final_selection_eval"
        (final / "selection.json").unlink()
        sheet(final, "post-epoch skill", dict.fromkeys("abcd", 1))
        (archive / "skills/skill_v0002.md").write_text("post-epoch skill")
        (archive / "best_skill.md").write_text("post-epoch skill")
        data = load_archive(archive, list("abcd"), population_cap=1)
        assert [(r["id"], r["correct"]) for r in data["population"]] == [("final", 4)]
        assert data["initial_population"]["excluded_by_cap"] == 1
        assert data["history"]["a"] == [0, 0]


    def test_repeated_full_skill_dedup_and_different_sheet_binding(self):
        archive = self.archive
        final = archive / "final_selection_eval"
        (final / "selection.json").unlink()
        sheet(final, "parent", dict.fromkeys("abcd", 1))
        audit = load_archive(archive, list("abcd"))["initial_population"]
        assert audit["unique_complete_skills"] == 1 and audit["complete_sheets"] == 2
        assert audit["records"][0]["canonical_id"] == "final"
        (archive / "skills/skill_v0002.md").write_text("another")
        with self.assertRaisesRegex(ValueError, "system prompt"):
            load_archive(archive, list("abcd"))


    def test_corrupt_prefix_or_missing_eval_rejected(self):
        archive = self.archive
        meta = archive / "history.json"
        bad = json.loads(meta.read_text())
        bad[0]["selection_early_stop"]["batches"][0]["hard_sum"] = 1
        write(meta, bad)
        with self.assertRaisesRegex(ValueError, "cumulative"):
            load_archive(archive, list("abcd"))
        (archive / "steps/step_0001/selection_eval/results.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "no answer sheet"):
            load_archive(archive, list("abcd"))

    def test_incomplete_run_rejected_and_completed_dev_only_supported(self):
        source = self.archive
        (source / "final_selection_eval/selection.json").unlink()
        with self.assertRaisesRegex(ValueError, "completed"):
            load_archive(source, list("abcd"))
        write(source / "config.json", {"env": "bird", "eval_test": False})
        write(source / "runtime_state.json", {"last_completed_step": 2})
        (source / "summary.json").write_text("contents must never be read")
        assert load_archive(source, list("abcd"))["initial_population"]["selected"] == ["v0"]
