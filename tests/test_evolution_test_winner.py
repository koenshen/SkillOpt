import fcntl
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evolution_test import run
from skillopt_evolution.usage import save_json


class FrozenWinnerTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.path = Path(t.name)
        (self.path / ".run.lock").touch()

    def publish(self):
        text = b"selected skill"
        (self.path / "best_skill.md").write_bytes(text)
        save_json(self.path / "summary.json", {"status": "complete", "best": {
            "sha256": hashlib.sha256(text).hexdigest(), "id": "final"}})

    def test_waits_for_active_evolution_and_uses_verified_skill(self):
        save_json(self.path / "summary.json", {"status": "running"})
        save_json(self.path / "last_failure.json", {"exception": "older attempt"})
        with (self.path / ".run.lock").open("a") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX)
            with patch.object(run.time, "sleep", side_effect=lambda _: self.publish()) as wait:
                state, text = run.wait_for_winner(self.path, 0)
        wait.assert_called_once()
        self.assertEqual(text, "selected skill")
        self.assertEqual(state["best"]["id"], "final")

    def test_stopped_run_and_mismatched_skill_cannot_trigger_test(self):
        with self.assertRaisesRegex(RuntimeError, "without a final"):
            run.wait_for_winner(self.path, 0)
        self.publish()
        (self.path / "best_skill.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            run.wait_for_winner(self.path, 0)

    def test_uses_stage_two_target_and_reuses_only_verified_completed_results(self):
        self.publish()
        src = self.path / "native"
        cfg = {"target_model": "native-model", "env": "searchqa"}
        save_json(src / "config.json", cfg)
        split = self.path / "data"
        items = [{"id": str(i), "question": "Q" + str(i), "answers": ["A"]} for i in range(1000)]
        save_json(split / "test/items.json", items)
        save_json(self.path / "manifest.json", {
            "benchmark": "triviaqa", "source": str(src), "config_sha256": run.fingerprint(cfg),
            "git_revision": "native-rev", "split_path": str(split / "val/items.json"),
            "source_config": {"target_model": "evolution-model", "target_backend": "openai_compatible",
                              "workers": 24, "max_completion_tokens": 131072}})
        def fake_test(*, adapter, items, skill, out_dir):
            self.assertEqual(skill, "selected skill")
            self.assertEqual(len(items), 1000)
            (out_dir / "results.jsonl").write_text("".join(
                json.dumps({"id": x["id"], "hard": 1, "soft": 1}) + "\n" for x in items))
            return {"status": "complete", "em": 1.0, "f1": 1.0}
        with patch.object(run, "archived_adapter") as make, patch.object(
                run, "run_test", side_effect=fake_test) as score:
            first = run.evaluate_run(self.path)
            self.assertEqual(first["target_model"], "evolution-model")
            self.assertEqual(make.call_args.args[0]["target_model"], "evolution-model")
            self.assertEqual(run.evaluate_run(self.path), first)
            score.assert_called_once()
            (self.path / "test/results.jsonl").write_text("damaged")
            with self.assertRaisesRegex(ValueError, "completed report"):
                run.evaluate_run(self.path)
