import tempfile
import unittest
from pathlib import Path

from skillopt_evolution.train_check import TrainCheck


class Answerer:
    workers = 24

    def __init__(self, confirm):
        self.confirm = confirm
        self.damage = None
        self.calls = []

    def rollout(self, items, skill, folder):
        ids = [item["id"] for item in items]
        self.calls.append((skill, ids))
        if skill == "child" and self.damage is None:
            self.damage = ids[4]
        return [dict(id=k, hard=int(not (skill == "child" and k == self.damage
                                         and ("repeat_child" not in folder or self.confirm))), agent_ok=True)
                for k in ids]


class TrainCheckTests(unittest.TestCase):
    def run_check(self, confirmed):
        train = [dict(id=str(i), question=f"Which city is host of event {i}?", answers=[f"town{i}"]) for i in range(25)]
        dev = [dict(id="dev", question="Which city was the host?", answers=["town0"])]
        train += [dict(id="duplicate", question=dev[0]["question"], answers=["anything"])]
        panel = TrainCheck(train, dev)
        adapter = Answerer(confirmed)
        with tempfile.TemporaryDirectory() as work:
            kwargs = dict(adapter=adapter, parent={"id":"X", "skill":"parent"},
                          proposal={"skill":"child", "shown_example_ids":["dev"],
                                    "transfer_cues":{"target":"host city", "risk":"events"}},
                          selection={"target_ids":["dev"], "risk_ids":[]}, out_dir=Path(work), seed=42)
            r = panel.check(**kwargs)
            before = len(adapter.calls)
            self.assertEqual(panel.check(**kwargs), r)
            self.assertEqual(before, len(adapter.calls))
            ids = sum(r["groups"].values(), [])
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 12)
            self.assertFalse({"0","dev","duplicate"} & set(ids))
            self.assertEqual(r["first_regressions"], [adapter.damage])
            self.assertEqual(r["passed"], not confirmed)
            self.assertEqual(r["confirmed_regressions"], [adapter.damage] if confirmed else [])
            self.assertFalse(any("test" in k for _, keys in adapter.calls for k in keys))

    def test_veto_repeated_regression(self):
        self.run_check(True)

    def test_do_not_veto_an_unconfirmed_flip(self):
        self.run_check(False)
