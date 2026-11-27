"""Regression tests for the reviewed English DE template on the actual matrix generation path."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from skillopt_evolution import evolve, minimal_prompt
from skillopt.optimizer.skill import SLOW_UPDATE_START, SLOW_UPDATE_END
from tests.test_evolution_evolve import record


class SharedRulePromptTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.parent = record("x", {"t": 0, "r": 1}, skill=(
            "Use the schedule.\nUse the units in the record.\n\n"
            f"{SLOW_UPDATE_START}\nPrior guidance.\n{SLOW_UPDATE_END}\n"))
        self.a = record("a", {"t": 1, "r": 0}, skill="Preserve a literal <prompt0> when present. Read completion times.")
        self.b = record("b", {"t": 1, "r": 1}, skill="Read the recorded completion time for a finished job.")
        self.plan = {"status": "selected", "reason": "matrix_shared_success", "donor_a": "a",
                     "donor_b": "b", "target_ids": ["t"], "risk_ids": ["r"]}
        self.child = self.parent["skill"].replace("Use the schedule.", "Use the schedule unless the question asks when the job finished.")

    def call(self, reply):
        def trace(rec, k):
            return {"user_prompt": f"## Context\nA literal <prompt2>. A maintenance log.\n\n## Question\nWhich time is {k}?",
                    "response": "<answer>" + rec["rows"][k]["predicted_answer"] + "</answer>"}
        with patch.object(evolve, "chat_optimizer", return_value=(reply, {})) as model:
            result = evolve.generate_offspring(
                parent=self.parent, best=self.parent, donor_a=self.a, donor_b=self.b,
                selection=self.plan, out_dir=self.root, max_completion_tokens=131072, trace_reader=trace)
        return result, model

    def test_exact_template_single_pass_and_full_child(self):
        self.assertEqual(minimal_prompt.template_digest(),
                         "2937d20272a7c68478b52b8afc02eaec7d8d3873f69c5c5b8a450a0d45350a4c")
        reply = "1. Compare the two instructions.\nFinal Prompt: <prompt>" + self.child + "</prompt>"
        proposal, model = self.call(reply)
        self.assertEqual(proposal["skill"], self.child)
        self.assertEqual(proposal["shown_example_ids"], ["t","r"])
        self.assertEqual(proposal["edits"][0]["target"], "Use the schedule.")
        self.assertEqual(proposal["transfer_cues"]["risk"], "Use the schedule.")
        args = model.call_args.kwargs
        self.assertEqual((args["system"], args["max_completion_tokens"]), ("", 131072))
        self.assertIn("Prompt 1: " + self.a["skill"], args["user"])
        self.assertIn("A literal <prompt2>.", args["user"])
        self.assertIn("Question to fix", args["user"])
        self.assertIn("first historical guide) — actual reply (accepted, score 1)", args["user"])
        self.assertTrue(args["user"].endswith("1.\n"))
        self.assertEqual(self.call(reply)[1].call_count, 0)

    def test_skip_and_protection(self):
        result, model = self.call("Analysis. <skip>All three already specify completion.</skip>")
        self.assertEqual(result["reason"], "skip_pair")
        model.assert_called_once()
        with self.assertRaisesRegex(ValueError, "protected"):
            minimal_prompt.parse_reply("<prompt>Guide without the protected block</prompt>", self.parent["skill"], [])

    def test_reject_unrelated_changes_and_malformed_child(self):
        for child in ("New first line.\nMiddle.\nNew last line.", "Completely different."):
            with self.assertRaises(ValueError):
                minimal_prompt.parse_reply("<prompt>" + child + "</prompt>", "First.\nMiddle.\nLast.", [])
        result, model = self.call("<prompt>unclosed output")
        self.assertEqual(result["reason"], "invalid_full_prompt")
        model.assert_called_once()
        attempt = self.root / "generation_attempts/01"
        self.assertTrue((attempt / "validation.json").exists())
