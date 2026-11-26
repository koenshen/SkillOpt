"""Frozen evidence selection and exact edit provenance in phase-two evolution."""
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skillopt_evolution import evolve


def record(uid, scores, parent_hash=None, skill=None):
    text = skill or f"Use the evidence collected by the {uid} document."
    return {
        "id": uid, "skill": text, "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "parent_sha256": parent_hash, "rows": {
            k: {"id": k, "agent_ok": v is not None, "hard": 0 if v is None else v}
            for k, v in scores.items()
        }, "edits": [{"index": 1, "op": "append", "target": "", "content": text}],
    }


class EvolutionDonorTests(unittest.TestCase):
    def setUp(self):
        self.x = record("parent", {"1": 0, "2": 1, "3": 0})
        self.best = record("best", {"1": 0, "2": 1, "3": 1})
        self.a = record("donor_a", {"1": 1, "2": 0}, "ancestral")
        self.b = record("donor_b", {"1": 0, "2": 1}, "ancestral")
        self.c = record("other", {"1": 1, "2": 1})
        self.rejected = [self.a, self.b, self.c]
        self.pop = [self.x, self.best]

    def select(self, pool=None, pop=None):
        return evolve.select_donors(parent=self.x, best=self.best,
                                    population=self.pop if pop is None else pop,
                                    rejected=self.rejected if pool is None else pool,
                                    mode="matrix", rng=random.Random(42))

    def test_observed_positive_siblings_and_counterexamples(self):
        s = self.select()
        self.assertEqual((s["donor_a"], s["donor_b"]), ("donor_a", "donor_b"))
        self.assertTrue(s["same_parent"])
        self.assertEqual(s["target_ids"], ["1"])
        self.assertEqual(s["risk_ids"], ["2"])
        self.assertEqual(s["distinct_support"], {"1": 2})

    def test_duplicate_text_and_failed_or_missing_scores_do_not_trigger(self):
        for other in (record("same", {"1": 1}, skill=self.a["skill"]),
                      record("failed", {"1": None}), record("missing", {})):
            s = self.select([self.a, self.b, other], [self.x])
            self.assertEqual(s["status"], "skipped")

    def test_no_observed_failure_in_controls(self):
        self.b["rows"].pop("1")
        # Restrict population to X, so the comparator cannot fall back on best.
        self.assertEqual(self.select(pop=[self.x])["status"], "skipped")

    def test_random_fallback_draws_distinct_population_documents(self):
        s = self.select(pool=[])
        self.assertEqual(s["reason"], "no_repeated_repair_random_fallback")
        self.assertEqual({s["donor_a"], s["donor_b"]}, {"parent", "best"})

    def test_generation_rejects_wrong_index_and_reuses_valid_result_on_resume(self):
        s = self.select()
        response = {
            "difference": "A uses a different source.",
            "mutation": "Keep useful distinctions.",
            "best_integration": "Keep the best rules as the base.",
            "crossover": "Transplant one rule onto X.",
            "hoped_for": "Preserving specific entities.",
            "at_risk": "Overlong answers.",
            "sources": [
                {"donor": "A", "edit_index": 1, "field": "content", "quote": self.a["skill"]},
                {"donor": "B", "edit_index": None, "field": "skill", "quote": self.b["skill"]},
            ],
            "edits": [{"op": "append", "content": "Use evidence tied to the requested entity."}],
        }
        wrong = json.loads(json.dumps(response))
        wrong["sources"][0]["edit_index"] = 2
        with tempfile.TemporaryDirectory() as folder:
            args = dict(parent=self.x, best=self.best, donor_a=self.a, donor_b=self.b,
                        selection=s, out_dir=Path(folder))
            with patch.object(evolve, "chat_optimizer", side_effect=[
                (json.dumps(wrong), {}), (json.dumps(response), {}),
            ]) as optimizer:
                result = evolve.generate_offspring(**args)
                resumed = evolve.generate_offspring(**args)
            self.assertEqual(optimizer.call_count, 2)
            self.assertEqual(result, resumed)
            self.assertEqual(result["edits"][0]["status"], "applied_append")
            self.assertEqual(result["status"], "generated")
            self.assertTrue((Path(folder)/"generation_attempts/01/validation.json").exists())
            self.assertEqual(len(list((Path(folder)/"token_events").glob("*.json"))), 2)
