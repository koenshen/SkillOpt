"""Matrix-mode donors are picked by their common, observed successes on X's mistakes."""
import random
import unittest

from skillopt_evolution.evolve import select_donors
from tests.test_evolution_evolve import record


class DualDonorTests(unittest.TestCase):
    def setUp(self):
        self.x = record("X", {"a": 0, "b": 0, "c": 0, "risk": 1, "unseen": 1})
        self.a = record("A", {"a": 1, "b": 1, "risk": 0, "unseen": 0})
        self.b = record("B", {"a": 1, "b": 1, "risk": 1})
        self.c = record("C", {"a": 1, "c": 1, "risk": 0})

    def choose(self, *pool, **overrides):
        return select_donors(parent=self.x, best=self.x, population=[self.x],
                             rejected=list(pool), rng=random.Random(7), mode="matrix", **overrides)

    def test_multi_question_pair_and_shared_observed_risk(self):
        plan = self.choose(self.a, self.b, self.c)
        self.assertEqual(plan["reason"], "matrix_shared_success")
        self.assertEqual(plan["target_ids"], ["a", "b"])
        self.assertEqual(plan["risk_ids"], ["risk"])
        self.assertEqual(plan["distinct_support"], {"a": 3, "b": 2})

    def test_retry_excludes_both_orientations_and_exhausts_real_pairs(self):
        tried = []
        for n in range(3):
            plan = self.choose(self.c, self.b, self.a, excluded_pairs=tried)
            self.assertEqual(plan["eligible_pairs"], 3 - n)
            tried.append(tuple(reversed(plan["donor_hashes"])))
        self.assertEqual(len({frozenset(pair) for pair in tried}), 3)
        self.assertEqual(self.choose(self.a, self.b, self.c, excluded_pairs=tried)["status"], "skipped")

    def test_prefer_an_unused_pair_across_slots(self):
        plan = self.choose(self.a, self.b, self.c, used_pairs=[("B", "A")])
        self.assertNotEqual({plan["donor_a"], plan["donor_b"]}, {"A", "B"})

    def test_inherited_rule_valid_even_without_an_applied_edit_in_that_step(self):
        self.a["edits"] = []
        self.b["edits"] = []
        self.assertEqual(self.choose(self.a, self.b)["status"], "selected")
