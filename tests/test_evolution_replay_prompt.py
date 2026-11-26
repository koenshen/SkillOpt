"""The meta-prompt replay must score the same historical item/slot opportunities."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillopt_evolution import replay_prompt as replay
from skillopt_evolution.replay_prompt import compare_examples
from skillopt_evolution.usage import save_json


def score(item_id, hard, answer, valid=True):
    return {"id": item_id, "hard": hard, "agent_ok": valid, "predicted_answer": answer}


class FrozenOpportunityTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": k, "kind": kind, "question": f"Which rule applies to {k}?", "answers": {}}
            for k, kind in (("fixed", "target"), ("missed", "target"), ("skipped_by_old", "target"),
                            ("old_api_failed", "risk"), ("risk", "risk"))
        ]
        self.old = {k: score(k, 0, "old prediction") for k in ["fixed", "missed", "risk"]}
        self.old["old_api_failed"] = score("old_api_failed", 0, "", False)
        self.parent = {e["id"]: score(e["id"], int(e["kind"] == "risk"), "parent prediction")
                       for e in self.examples}

    def test_new_child_uses_only_questions_that_old_child_actually_answered(self):
        new = {k: score(k, int(k != "missed"), "new prediction") for k in self.parent}
        result = compare_examples(self.examples, self.old, new, self.parent, skipped=False)
        self.assertEqual(result["target"]["n"], 2)
        self.assertEqual(result["target"]["old_hits"], 0)
        self.assertEqual(result["target"]["new_hits"], 1)
        self.assertEqual(result["target"]["new_measured"], 2)
        self.assertEqual(result["target"]["fallback_to_parent"], 0)
        self.assertEqual((result["risk"]["n"], result["risk"]["new_hits"]), (1, 1))
        self.assertEqual({r["id"] for r in result["target"]["rows"]}, {"fixed", "missed"})

    def test_skipped_pair_reports_unchanged_parent_without_counting_a_model_answer(self):
        result = compare_examples(self.examples, self.old, {}, self.parent, skipped=True)
        for role, n, expected in [("target", 2, 0), ("risk", 1, 1)]:
            with self.subTest(role=role):
                block = result[role]
                self.assertEqual((block["n"], block["new_hits"]), (n, expected))
                self.assertEqual((block["new_measured"], block["fallback_to_parent"]), (0, n))

class ProbeIsolationTests(unittest.TestCase):
    def test_probes_only_missing_comparable_examples_and_never_reopens_a_rejected_gate(self):
        ids = ["k0", "k1", "k2", "k3"]
        items = [{"id": k, "question": k} for k in ids]

        def record(uid, outcomes):
            text = "Only rule for " + uid
            rows = {k: dict(score(k, h, f"answer {uid} {k}"), question=k)
                    for k, h in zip(ids, outcomes)}
            return {"id": uid, "skill": text, "sha256": replay.sha(text), "rows": rows,
                    "edits": [], "complete": True, "correct": sum(outcomes)}

        x, a, b = record("x", [0, 1, 0, 1]), record("a", [1, 0, 1, 0]), record("b", [0, 1, 0, 1])
        plan = {"parent": "x", "best": "x", "donor_a": "a", "donor_b": "b",
                "target_ids": ["k2"], "risk_ids": ["k3"], "reason": "matrix_observed_contrast"}
        roles = {"X": x, "best": x, "A": a, "B": b}
        original = {"edit_budget": 2, "selection_mode": plan["reason"],
                    "role_to_document": {k: d["sha256"] for k, d in roles.items()},
                    "documents": {d["sha256"]: d["skill"] for d in roles.values()},
                    "applied_edits": {"A": [], "B": []},
                    "paired_examples": replay._examples(x, x, a, b, plan)}
        score_calls = []

        class Adapter:
            max_completion_tokens = 131072
            def rollout(self, items, skill, out):
                keys = [i["id"] for i in items]
                score_calls.append(keys)
                return [score(k, int(k in ("k2", "k3")), k) for k in keys]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            old = base / "old/slot_01"
            save_json(old / "generation_attempts/01/request.json", {"user": json.dumps(original)})
            old_rows = {k: score(k, 0, k) for k in ids}
            save_json(old / "decision.json", {"parent": "x", "parent_correct": 2, "decision": "bound_reject",
                                               "rows": old_rows, "sha256": "archived"})
            save_json(old / "evaluation_inputs.json", {"signature": {"frozen_order": ids, "child_sha256": "archived",
                                                                       "parent_correct": 2, "batch_size": 2}})
            new = {"status": "generated", "skill": "A different trial rule.", "edits": []}
            with patch.object(replay, "generate_offspring", return_value=new) as generate:
                args = dict(adapter=Adapter(), items=items, records={v["id"]:v for v in roles.values()},
                            generation_dir=old.parent, old_slot=old, work=base / "new", plan=plan,
                            frozen={"order": ids}, edit_budget=2, reserved_hashes=set())
                result = replay.replay_slot(**args)
                repeated = replay.replay_slot(**args)
            self.assertEqual(score_calls, [["k0", "k1"], ["k2", "k3"]])
            generate.assert_called_once()
            self.assertFalse(result["gate"]["accepted"])
            self.assertEqual(result["gate"]["n_observed"], 2)
            self.assertEqual(result["gate"]["upper_bound"], .5)
            self.assertEqual(result["comparison"]["target"]["new_hits"], 1)
            self.assertEqual(result["comparison"]["risk"]["new_hits"], 1)
            self.assertEqual(result, repeated)
