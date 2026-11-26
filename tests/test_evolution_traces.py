import json
import tempfile
import unittest
from pathlib import Path

from skillopt_evolution.traces import TraceReader


class TraceReaderTests(unittest.TestCase):
    def test_reads_served_prompt_and_visible_messages_without_grader_commentary(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sheet = "steps/step_0001/selection_eval/results.jsonl"
            pred = (base / sheet).parent / "predictions/q1"
            pred.mkdir(parents=True)
            (pred / "target_user_prompt.txt").write_text("## Context\nA source excerpt.\n## Question\nWhich place?")
            (pred / "conversation.json").write_text(json.dumps([
                {"type": "message", "turn": 1, "content": "<answer>Place</answer>"},
                {"role": "system", "content": "[EVALUATION RESULT] secret grader note"},
            ]))
            raw = {"id": "q1", "hard": 1, "agent_ok": True}
            origin = {"id": "native", "sha256": "abc", "sheet": sheet, "rows": {"q1": raw}}
            later = {"id": "later", "sha256": "abc", "row_sources": {"q1": "native"}, "rows": {"q1": raw}}
            reader = TraceReader(base, base / "evo")
            reader.register(origin)
            self.assertEqual(reader(later, "q1"), {
                "user_prompt": "## Context\nA source excerpt.\n## Question\nWhich place?",
                "response": "<answer>Place</answer>",
            })
            later["sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "does not match"):
                reader(later, "q1")

    def test_reads_second_stage_attempt_and_verifies_last_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work = base / "generations/001/slot_01"
            attempt = "evaluation_attempts/00003"
            pred = work / attempt / "rollout/predictions/q1"
            pred.mkdir(parents=True)
            (pred / "target_user_prompt.txt").write_text("An actual question.")
            (pred / "conversation.json").write_text(json.dumps([{"type": "message", "content": "Reply"}]))
            raw = {"agent_ok": True, "hard": 1, "response": "Reply"}
            rec = {"id": "g1", "source": str(work), "rows": {"q1": raw}, "row_sources": {"q1": attempt}}
            reader = TraceReader(None, base)
            self.assertEqual(reader(rec, "q1")["response"], "Reply")
            raw["response"] = "another response"
            with self.assertRaisesRegex(ValueError, "disagree"):
                reader(rec, "q1")
