"""Regression tests for the reviewed English DE template on the actual matrix generation path."""
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

    def test_only_approved_template_and_program_assembles_child(self):
        reply = "<old>Use the schedule.</old> → <new>Use the schedule unless the question asks when the job finished.</new>"
        proposal, model = self.call(reply)
        self.assertEqual(proposal["skill"], self.child)
        self.assertEqual(proposal["shown_example_ids"], ["t", "r"])
        args = model.call_args.kwargs
        self.assertEqual((args["system"], args["max_completion_tokens"]), ("", 131072))
        self.assertIn("[EDITABLE] Use the schedule.", args["user"])
        self.assertIn("[READ ONLY] " + SLOW_UPDATE_START, args["user"])
        self.assertIn("Prompt 1: " + self.a["skill"], args["user"])
        self.assertIn("A literal <prompt2>.", args["user"])
        self.assertIn("first historical guide) — actual reply (accepted, score 1)", args["user"])
        self.assertEqual(self.call(reply)[1].call_count, 0)

    def test_skip_and_reject_readonly_or_ambiguous_original(self):
        self.assertEqual(self.call("<skip>Both guides already use the same rule.</skip>")[0]["reason"], "skip_pair")
        original = self.parent["skill"] + "# A heading\nRepeated. Repeated."
        for target in ("Repeated.", "Prior guidance.", "# A heading", "the schedule"):
            with self.assertRaisesRegex(ValueError, "EDITABLE"):
                minimal_prompt.parse_reply(f"<old>{target}</old> → <new>Changed.</new>", original, [])

    def test_displayed_correct_and_incorrect_answers_cannot_be_newly_embedded(self):
        cases = [{"id":"q1", "question":"Where was the artist born?", "answers":{
            "A":{"answer":"Gary, Indiana"}, "B":{"answer":"The Barretts"}, "X":{"answer":"Face"}}}]
        for new in ("Output GARY — INDIANA.", "Use Barretts.", "For example, face."):
            p = minimal_prompt.parse_reply(f"<old>Use the schedule.</old> → <new>{new}</new>",
                                           self.parent["skill"], cases)
            self.assertEqual(p["reason"], "copied_displayed_answer")
            self.assertEqual(p["matches"][0]["id"], "q1")
        # An existing mention can be retained; an unrelated whole word is safe.
        self.assertEqual(minimal_prompt.parse_reply("Use Face. → Use Face where applicable.",
                         "Use Face.", cases)["status"], "generated")
        self.assertEqual(minimal_prompt.parse_reply("Use nouns. → Consider surface features.",
                         "Use nouns.", cases)["status"], "generated")

    def test_no_full_guides_or_multiple_edits(self):
        for reply in ["<prompt>an entire document</prompt>",
                      "<old>Use nouns.</old> → <new>Use names. Strip all titles.</new>",
                      "<old>Use nouns.</old> → <new>Use titles " + SLOW_UPDATE_START + "</new>",
                      "<old>Use nouns.</old> → <new>Use titles.</new> and something else"]:
            with self.assertRaises(ValueError):
                minimal_prompt.parse_reply(reply, "Use nouns.", [])

    def test_exact_offsets_and_punctuation_in_examples(self):
        parent = "  - Use `e.g. an example`. Keep other text.  \r\n"
        r = minimal_prompt.parse_reply("<old>Keep other text.</old> → <new>Keep the units.</new>", parent, [])
        self.assertEqual(r["skill"], "  - Use `e.g. an example`. Keep the units.  \r\n")
