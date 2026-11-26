import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evolution_test import scoring
from skillopt_evolution.usage import read_json, save_json


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.items = [{"id": str(i)} for i in range(4)]
        self.calls = []
        self.fail = {"1"}
        self.workers = 3
        self.costs = dict(calls=0, prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def rollout(self, items, skill, out):
        ids = [i["id"] for i in items]
        self.calls.append(ids)
        self.costs = {k: v+len(ids)*x for (k,v),x in zip(self.costs.items(), (1, 7, 3, 10))}
        rows = [dict(id=k, agent_ok=k not in self.fail,
                     hard=int(k in {"0", "1"} and k not in self.fail),
                     soft=(1.0 if k in {"0","1"} else .25) if k not in self.fail else 0.0)
                for k in ids]
        (Path(out)/"results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        return rows

    def score(self):
        with patch.object(scoring, "get_token_summary", side_effect=lambda: {"_total":self.costs.copy()}):
            return scoring.run_test(items=self.items, adapter=self, skill="frozen", out_dir=self.root)

    def test_full_denominator_and_only_unsuccessful_rows_retried_on_resume(self):
        r = self.score()
        self.assertEqual(self.calls, [["0", "1", "2"], ["1"], ["3"]])
        self.assertEqual((r["status"], r["n_answered"], r["n_correct"], r["failed_ids"]),
                         ("failed_items", 3, 1, ["1"]))
        self.assertEqual((r["em"],r["f1"]), (.25,.375))
        self.fail.clear()
        r = self.score()
        self.assertEqual(self.calls[-1], ["1"])
        self.assertEqual((r["status"],r["em"],r["f1"]),("complete",.5,.625))
        self.assertEqual(r["usage"]["input_tokens"],42)
        self.assertEqual(r["usage"]["output_tokens"],18)
        self.assertEqual(r["usage"]["calls"],6)
        self.assertEqual(self.score(),r)
        self.assertEqual(len(self.calls),4)
        rows = list(scoring._rows(self.root/"results.jsonl"))
        self.assertEqual([r["id"] for r in rows], ["0","1","2","3"])
        self.assertFalse(list(self.root.rglob("token_events/*.json")))

    def test_interrupted_batch_reuses_success_and_keeps_unknown_cost(self):
        p = self.root/"attempts"/"00001"
        save_json(p/"request.json",{"ids":["0","1"]})
        (p/"results.jsonl").write_text('{"id":"0","agent_ok":true,"hard":1,"soft":1.0}\n{"id":"1"')
        (p.parent / "00002").mkdir()  # Interrupted after mkdir, before recording a request.
        self.fail.clear()
        r=self.score()
        self.assertEqual(self.calls,[["1","2","3"]])
        self.assertEqual(r["usage"]["unfinished_attempts"],["00001"])
        self.assertEqual(read_json(self.root/"usage.json"),r["usage"])
        self.assertEqual(r["n_total"],4)
        self.assertTrue((p.parent / "00003" / "usage.json").is_file())
