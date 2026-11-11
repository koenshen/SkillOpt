"""Exactness tests for lineage-ordered selection early stopping."""
import random

from skillopt.evaluation.early_stop import (
    evaluate_with_early_stop,
    lineage_order,
    update_history,
)
from skillopt.utils import compute_score


def _runner(truth, calls):
    done = {}

    def run(prefix):
        for it in prefix:
            if it["id"] not in done:
                calls.append(it["id"])
                done[it["id"]] = {"id": it["id"], "hard": truth[it["id"]], "soft": 0.0}
        return list(done.values())

    return run


def test_lineage_order_lowest_first_unseen_as_zero():
    hist = {"a": [1, 1], "b": [0, 1], "c": [0, 0]}
    assert lineage_order(["a", "b", "c", "d"], hist) == ["c", "d", "b", "a"]


def test_decision_matches_full_eval_randomized():
    rng = random.Random(0)
    for trial in range(500):
        n = rng.choice([17, 24, 50, 200])
        items = [{"id": f"q{i}"} for i in range(n)]
        truth = {it["id"]: rng.randint(0, 1) for it in items}
        current = rng.randint(0, n) / n
        hist = {}
        for _ in range(rng.randint(0, 4)):
            update_history(hist, [{"id": i, "hard": rng.randint(0, 1)} for i in truth])
        calls = []
        out = evaluate_with_early_stop(
            items, hist, _runner(truth, calls), current, rng.choice([1, 8, 24]))
        full_hard = sum(truth.values()) / n
        full_accept = full_hard > current
        es_accept = (not out.stopped) and compute_score(out.results)[0] > current
        assert es_accept == full_accept
        assert len(calls) == len(set(calls)) <= n          # each item at most once
        assert out.lower_bound <= full_hard <= out.upper_bound + 1e-12
        if full_accept:
            assert not out.stopped and out.n_evaluated == n
            assert compute_score(out.results)[0] == full_hard
        if out.stopped:
            assert out.upper_bound <= current


def test_stops_early_when_hard_items_fail():
    items = [{"id": f"q{i}"} for i in range(200)]
    truth = {it["id"]: (1 if i >= 60 else 0) for i, it in enumerate(items)}
    hist = {f"q{i}": [0.0] if i < 60 else [1.0] for i in range(200)}
    calls = []
    out = evaluate_with_early_stop(items, hist, _runner(truth, calls), 140 / 200, 24)
    assert out.stopped and out.n_evaluated == 72 and len(calls) == 72
