"""A generation never counts empty, repeated, or unassessed offspring slots."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillopt_evolution import run, usage


def record(uid, scores, action):
    text = f"Read the evidence using rule {uid}."
    return dict(id=uid, skill=text, sha256=run.sha(text), parent_sha256=None, action=action,
                rows={str(i): {"id": str(i), "agent_ok": True, "hard": v} for i, v in enumerate(scores)},
                edits=[], complete=True, correct=sum(scores), score=sum(scores) / len(scores))


class LocalAdapter:
    max_completion_tokens = 131072

    def __init__(self):
        self.evaluations = []

    def rollout(self, items, skill, folder):
        self.evaluations.append(skill)
        return [dict(item, hard=0, agent_ok=True) for item in items]


class FillGenerationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        self.items = [{"id": str(i), "question": f"Question {i}"} for i in range(4)]
        self.parents = [record(f"P{i}", [0, 1, 0, 1], "accepted") for i in range(5)]
        self.donors = [record(f"D{i}", [1, 0, 1, 1], "reject") for i in range(5)]
        self.adapter = LocalAdapter()
        self.calls = []

    def run_generation(self, donors, generate, checker=None):
        entries = [*self.parents, *donors]
        archive = dict(population=self.parents, records=entries, skipped=[],
                       initial_population={}, native_step_usage=dict(input_tokens=0, output_tokens=0, total_tokens=0),
                       history={str(i): [r["rows"][str(i)]["hard"] for r in entries] for i in range(4)})
        with patch.object(run, "generate_offspring", side_effect=generate):
            return run.run_loop(archive=archive, items=self.items, adapter=self.adapter, out=self.out, train_checker=checker,
                                options=dict(seed=42, generations=1, patience=2, strategy="matrix", edit_budget=2))

    def test_refills_skipped_and_duplicate_slots_before_any_validation(self):
        def generate(*, parent, donor_a, donor_b, **kwargs):
            self.assertEqual(self.adapter.evaluations, [])
            pair = frozenset((donor_a["sha256"], donor_b["sha256"]))
            self.calls.append((parent["id"], pair))
            count = sum(uid == parent["id"] for uid, _ in self.calls)
            if parent == self.parents[0] and count == 1:
                return {"status": "skipped", "reason": "unexplained_contrast"}
            if parent == self.parents[0] and count == 2:
                text = self.donors[0]["skill"]  # Previously rejected outside the population.
            elif parent == self.parents[1] and count == 1:
                text = "Child of P0"  # This generation already has that child.
            else:
                text = f"Child of {parent['id']}"
            return dict(status="generated", skill=text, edits=[], mutation=text)
        result = self.run_generation(self.donors, generate)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(set(self.adapter.evaluations)), 5)
        self.assertEqual(len(self.calls), 8)
        self.assertEqual(len(self.calls), len(set(self.calls)))
        self.assertEqual(result["generations"][0]["generated"], 5)
        self.assertEqual(result["generations"][0]["evaluated"], 5)
        frozen = usage.read_json(self.out / "generations/001/frozen.json")
        self.assertEqual([p["donor_attempt"] for p in frozen["plans"]], [3, 2, 1, 1, 1])
        first = usage.read_json(self.out / "generations/001/slot_01/donor_attempts.json")
        self.assertEqual([p["status"] for p in first], ["skipped", "skipped", "generated"])
        calls_before = list(self.adapter.evaluations)
        self.assertEqual(self.run_generation(self.donors, self.fail), result)
        self.assertEqual(self.adapter.evaluations, calls_before)

    def test_exhaustion_is_not_a_completed_generation_or_plateau(self):
        def skip(**kwargs):
            self.calls.append(kwargs["parent"]["id"])
            return {"status": "skipped", "reason": "unexplained_contrast"}
        with self.assertRaisesRegex(RuntimeError, "unfilled"):
            self.run_generation(self.donors[:2], skip)
        self.assertEqual(self.calls, ["P0"])
        self.assertFalse(self.adapter.evaluations)
        status = usage.read_json(self.out / "summary.json")
        self.assertEqual(status["status"], "incomplete")
        self.assertEqual((status["required"], status["generated"], status["evaluated"]), (5, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "unfilled"):
            self.run_generation(self.donors[:2], self.fail)

    def test_failed_training_guard_refills_before_dev_and_keeps_audit(self):
        from unittest.mock import Mock
        checker = Mock()
        checker.check.side_effect = [{"passed":False, "confirmed_regressions":["train-1"]}] + [{"passed":True}]*5
        def generate(**kwargs):
            text = f"child {len(self.calls)}"
            self.calls.append(text)
            return {"status":"generated", "skill":text, "edits":[], "mutation":text}
        result = self.run_generation(self.donors, generate, checker)
        self.assertEqual(result["generations"][0]["evaluated"], 5)
        self.assertEqual(len(self.calls), 6)
        self.assertNotIn(self.calls[0], self.adapter.evaluations)
        attempt = usage.read_json(self.out / "generations/001/slot_01/donor_attempts/01/proposal.json")
        self.assertEqual(attempt["reason"], "confirmed_train_regression")
        self.assertEqual(attempt["train_check"]["confirmed_regressions"], ["train-1"])
