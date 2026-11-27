"""Use a true two-success contrast and only evidence visible at inference time."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillopt_evolution import evolve
from tests.test_evolution_evolve import record


class SharedRulePromptTests(unittest.TestCase):
    def setUp(self):
        self.x = record("x", {"t": 0, "r": 1}, skill="Look at the scheduled time first.")
        self.a = record("a", {"t": 1, "r": 0}, skill="Read the actual recorded time of an event.")
        self.b = record("b", {"t": 1, "r": 1}, skill="Use the completion entry for actual events.")
        self.plan = {"status": "selected", "reason": "matrix_shared_success", "donor_a": "a",
                     "donor_b": "b", "target_ids": ["t"], "risk_ids": ["r"]}
        self.b["rows"]["t"]["predicted_answer"] = "different accepted alias"
        self.patch = {"decision": "propose",
                      "diagnosis": {"kind": "missing_step", "target_id": "t", "accepted_answer": "answer-1",
                                    "x_answer": "answer-0", "donor_action": "Read completion entry.",
                                    "x_action": "Read the schedule."},
                      "transfer_cues": {"target": "finished events", "risk": "scheduled events"}, "common_rule": "Select completed observations.",
                      "difference": "X chooses the schedule instead.", "x_rule_quote": self.x["skill"], "mutation": "Restrict to actual event.",
                      "best_integration": "Keep schedule interpretation for plans.",
                      "crossover": "Insert branch on completed vs planned.",
                      "hoped_for": "Find actual times", "at_risk": "Confuse a forecast with completion",
                      "sources": [{"donor": role, "edit_index": None, "field": "skill", "quote": donor["skill"]}
                                  for role, donor in (("A", self.a), ("B", self.b))],
                      "edits": [{"op": "insert_after", "target": self.x["skill"], "content": "For a finished event select its completion entry."}],
                      "checks": [{"id": k, "outcome": outcome, "passage_quote": "Maintenance log",
                                  "application": "Identify the source record for the requested event.",
                                  "patch_edit": 1} for k, outcome in (("t", "repair"), ("r", "preserve"))]}

    @staticmethod
    def trace(rec, k):
        return {"user_prompt": "## Context\nMaintenance log: schedule 1, recorded 2.\n\n## Question\nQuestion " + k + "?",
                "response": "<answer>" + rec["rows"][k]["predicted_answer"] + "</answer>"}

    def propose(self, responses):
        with tempfile.TemporaryDirectory() as tmp, patch.object(evolve, "chat_optimizer",
                 side_effect=[(json.dumps(x), {}) for x in responses]) as chat:
            p = evolve.generate_offspring(parent=self.x, best=self.x, donor_a=self.a, donor_b=self.b,
                                          selection=self.plan, out_dir=Path(tmp), max_completion_tokens=131072,
                                          trace_reader=self.trace)
            return p, chat.call_args_list

    def test_reflects_on_both_accepted_visible_responses_and_one_exact_input(self):
        p, calls = self.propose([self.patch])
        self.assertEqual(p["status"], "generated")
        self.assertEqual(p["common_rule"], self.patch["common_rule"])
        example = json.loads(calls[0].kwargs["user"])["paired_examples"][0]
        self.assertEqual(len(example["served_inputs"]), 1)
        for role in ("A", "B"):
            self.assertEqual(example["answers"][role]["verdict"], "accepted")
        self.assertEqual(example["answers"]["B"]["visible_response"], "<answer>different accepted alias</answer>")
        self.assertIn("Maintenance log", next(iter(example["served_inputs"].values())))

    def test_wrong_citation_from_question_retried(self):
        bad = json.loads(json.dumps(self.patch))
        bad["checks"][0]["passage_quote"] = "Question t?"
        p, calls = self.propose([bad, self.patch])
        self.assertEqual(p["status"], "generated")
        self.assertEqual(len(calls), 2)
        self.assertIn("passage_quote", calls[1].kwargs["user"])

    def test_no_shared_rule_requests_new_donor_pair(self):
        p, calls = self.propose([{"decision": "skip_pair", "edits": [], "checked_target_ids": ["t"],
                                  "diagnosis": {"kind":"unsupported"}, "reason": "The two accepted donors specify unrelated rules."}])
        self.assertEqual(p["status"], "skipped")
        self.assertEqual(len(calls), 1)

    def test_conflict_replaces_only_the_broad_sentence(self):
        keep = "Read the completion entry for finished events."
        broad = "Use planned times for every inspection."
        self.x["skill"] = f"{keep} {broad} Preserve other information."
        doc = self.patch
        doc["diagnosis"].update(kind="conflict", keep_rule_quote=keep, keep_action="This job finished, so use actual time.")
        doc["x_rule_quote"] = broad
        content = "Use the planned time for an inspection only when asked for its schedule."
        doc["edits"] = [{"op":"replace", "target":broad, "content":content}]
        corrupt = json.loads(json.dumps(doc))
        corrupt["edits"][0]["target"] = keep
        bad_answer = json.loads(json.dumps(doc))
        bad_answer["diagnosis"]["x_answer"] = doc["diagnosis"]["accepted_answer"]
        proposal, calls = self.propose([corrupt, bad_answer, doc])
        self.assertEqual(len(calls), 3)
        self.assertEqual(proposal["skill"], self.x["skill"].replace(broad, content))

    def test_equivalent_citations_in_x_trigger_new_donors(self):
        original = self.x["skill"]
        self.x["skill"] += " " + self.a["skill"] + " " + self.b["skill"]
        self.patch["x_rule_quote"] = original
        proposal, calls = self.propose([self.patch])
        self.assertEqual(proposal["reason"], "existing_rule")
        self.assertEqual(len(calls), 1)
