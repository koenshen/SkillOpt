import json
from pathlib import Path

import tempfile
import unittest
from unittest.mock import patch

from skillopt_evolution.evaluate import evaluate_candidate
from skillopt_evolution.usage import read_json, summarize_usage


def sheet(truth):
    return {str(i): {"id": str(i), "hard": hit, "agent_ok": True} for i, hit in enumerate(truth)}


class Adapter:
    def __init__(self, truth, fail=False):
        self.rows = sheet(truth)
        self.calls = []
        self.fail = fail

    def rollout(self, items, skill, dest):
        keys = [item["id"] for item in items]
        self.calls.append(keys)
        return [dict(self.rows[k], agent_ok=not (self.fail and k == "2")) for k in keys]


def evaluate(directory, adapter, parent, cache=None, order=None, batch_size=4):
    return evaluate_candidate(
        adapter=adapter, skill="new skill", dev_items=[{"id": k} for k in parent],
        order=order or list(parent), parent_rows=parent, cached_rows=cache or {},
        out_dir=directory, batch_size=batch_size,
    )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)

    def test_extend_cached_prefix_for_lower_parent_and_do_not_repeat_old_answers(self):
        tmp_path = self.directory
        adapter = Adapter([0, 0, 0, 0, 1, 1, 1, 1])
        high = sheet([1] * 4 + [0] * 4)
        first = evaluate(tmp_path / "s1", adapter, high)
        assert first["decision"] == "bound_reject"
        assert first["accuracy"] is None and first["upper_bound"] == 0.5
        assert first["regressions"] == ["0", "1", "2", "3"]
        assert len(adapter.calls) == 1

        # Same rejected child already cannot beat a higher parent; no API call.
        high["4"]["hard"] = 1
        other = evaluate(tmp_path / "s2", adapter, high, first["rows"])
        assert not other["accepted"] and not other["new_ids"] and len(adapter.calls) == 1

        # A lower parent plus a different generation order needs the rest of the sheet.
        low = sheet([1, 0, 0, 0, 0, 0, 0, 0])
        later = evaluate(tmp_path / "s3", adapter, low, first["rows"], order=list(reversed(low)))
        assert later["accepted"] and later["accuracy"] == 0.5
        assert later["new_ids"] == ["7", "6", "5", "4"]
        assert later["fixes"] == ["7", "6", "5", "4"] and later["regressions"] == ["0"]
        assert adapter.calls == [["0", "1", "2", "3"], ["7", "6", "5", "4"]]

        # Same slot/resume must not re-charge, even with a later expanded external cache.
        assert evaluate(tmp_path / "s3", adapter, low, later["rows"], order=list(reversed(low))) == later
        assert len(adapter.calls) == 2


    def test_failures_are_retried_and_do_not_reject_candidate(self):
        tmp_path = self.directory
        parent = sheet([1, 1, 1, 0])
        adapter = Adapter([1, 1, 1, 1], fail=True)
        with self.assertRaisesRegex(RuntimeError, "Unresolved dev requests"):
            evaluate(tmp_path, adapter, parent)
        assert adapter.calls == [list(parent), ["2"]]
        assert not (tmp_path / "evaluation.json").exists()
        assert len(list((tmp_path / "token_events").glob("*.json"))) == 2
        adapter.fail = False
        result = evaluate(tmp_path, adapter, parent)
        assert result["accepted"] and result["correct"] == 4 and adapter.calls[-1] == ["2"]
        assert result["new_ids"] == list(parent)
        for path in sorted((tmp_path / "evaluation_attempts").glob("*/request.json")):
            assert (tmp_path / read_json(path)["token_event"]).exists()


    def test_interrupted_call_restores_successes_only(self):
        tmp_path = self.directory
        class Interrupted(Adapter):
            def rollout(self, items, skill, dest):
                path = Path(dest)
                path.mkdir(parents=True)
                (path / "results.jsonl").write_text(json.dumps(self.rows["0"]) + '\n{"id":', encoding="utf-8")
                raise KeyboardInterrupt()

        parent = sheet([0, 1, 0, 0])
        with self.assertRaises(KeyboardInterrupt):
            evaluate(tmp_path, Interrupted([1, 1, 1, 1]), parent)
        recovery = Adapter([1, 1, 1, 1])
        result = evaluate(tmp_path, recovery, parent)
        assert result["accepted"] and result["correct"] == 4 and recovery.calls == [["1", "2", "3"]]
        assert summarize_usage(tmp_path)["failed_events"] == 1
        with self.assertRaisesRegex(ValueError, "Resume inputs changed"):
            evaluate(tmp_path, recovery, parent, order=list(reversed(parent)))


    def test_complete_cache_tie_and_invalid_archived_row(self):
        tmp_path = self.directory
        adapter = Adapter([1, 0, 1, 0])
        parent = sheet([1, 1, 0, 0])
        cache = sheet([1, 0, 1, 0])
        result = evaluate(tmp_path / "tie", adapter, parent, cache)
        assert result["complete"] and not result["accepted"] and result["decision"] == "full_no_win"
        assert not adapter.calls
        cache["2"]["agent_ok"] = False
        parent["0"]["hard"] = 0
        result = evaluate(tmp_path / "retry", adapter, parent, cache)
        assert result["recovered_failure_ids"] == ["2"]
        assert adapter.calls == [["2"]]

    def test_token_events_link_to_batch_and_skip_cached_items(self):
        adapter = Adapter([1, 0, 0, 0])
        parent = sheet([0, 0, 0, 0])
        totals = lambda: {"_total": dict(calls=len(adapter.calls),
                                          prompt_tokens=11 * len(adapter.calls),
                                          completion_tokens=2 * len(adapter.calls),
                                          total_tokens=13 * len(adapter.calls))}
        with patch("skillopt_evolution.usage.get_token_summary", totals):
            result = evaluate(self.directory, adapter, parent, batch_size=2)
            assert result["complete"]
            assert summarize_usage(self.directory)["validation"] == {
                "calls": 2, "input_tokens": 22, "output_tokens": 4, "total_tokens": 26,
            }
