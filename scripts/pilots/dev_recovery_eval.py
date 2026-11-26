"""Dev-only screening and paired confirmation for the recovery pilot."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from skillopt.evaluation.early_stop import evaluate_with_early_stop, lineage_order
from skillopt.model import get_token_summary


def _write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _index(rows):
    indexed = {str(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate dev item ID")
    return indexed


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@contextmanager
def _measure(usage_path, stage):
    # The model tracker is process-wide. Measure across the entire parallel
    # screen so that one request is charged exactly once.
    before = get_token_summary()["_total"]
    try:
        yield
    finally:
        after = get_token_summary()["_total"]
        costs = json.loads(usage_path.read_text()) if usage_path.exists() else {}
        totals = costs.setdefault(stage, {})
        for key, value in after.items():
            totals[key] = totals.get(key, 0) + value - before[key]
        _write(usage_path, costs)


def _answers(adapter, items, skill, directory):
    """Resume one score sheet; isolate failed-call retries from original rows."""
    raw_file = directory / "results.jsonl"
    rows = {}
    if raw_file.exists():
        with raw_file.open() as f:
            rows = _index([json.loads(line) for line in f if line.strip()])
    wanted = [str(item["id"]) for item in items]
    if any(key not in rows for key in wanted):
        rows = _index(adapter.rollout(items, skill, str(directory)))
    if any(key not in rows for key in wanted):
        raise RuntimeError(f"Adapter omitted requested dev answers in {directory}")
    if any(not rows[key]["agent_ok"] for key in wanted):
        retries = sorted(directory.glob("retry_*"))
        for retry in retries:
            path = retry / "results.jsonl"
            if path.exists():
                with path.open() as f:
                    for line in f:
                        answer = json.loads(line)
                        if answer["agent_ok"]:
                            rows[str(answer["id"])] = answer
        failed = [item for item in items if not rows[str(item["id"])]["agent_ok"]]
        if failed:
            retry = directory / f"retry_{len(retries)+1:03d}"
            for answer in adapter.rollout(failed, skill, str(retry)):
                rows[str(answer["id"])] = answer
    if any(not rows[key]["agent_ok"] for key in wanted):
        raise RuntimeError(f"Dev execution failed after retry in {directory}")
    return [rows[key] for key in wanted]


def _compare(results, reference, evidence):
    fixed, lost = [], []
    for row in results:
        key = str(row["id"])
        if row["hard"] > reference[key]["hard"]:
            fixed.append(key)
        elif row["hard"] < reference[key]["hard"]:
            lost.append(key)
    return {
        "observed": len(results),
        "fixes": fixed,
        "regressions": lost,
        "net_observed": len(fixed) - len(lost),
        "outside_evidence_observed": sum(str(row["id"]) not in evidence for row in results),
        "outside_evidence_fixes": sorted(set(fixed) - evidence),
        "outside_evidence_regressions": sorted(set(lost) - evidence),
        "reference_execution_failures_observed": sum(
            not reference[str(row["id"])]["agent_ok"] for row in results
        ),
    }


def evaluate_candidates(*, adapter, dev_items, history, anchor_skill, anchor_rows,
                        candidates, out_dir, evidence_ids=()):
    """Screen two candidates per arm against one frozen, complete dev anchor."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    reference = _index(anchor_rows)
    dev_by_id = _index(dev_items)
    ids = list(dev_by_id)
    if len(ids) != 200 or set(reference) != set(ids) or set(history) != set(ids):
        raise ValueError("Expected matching complete 200-item dev reference, split, and history")
    if set(candidates) != {"ordinary", "recovery"} or any(len(v) != 2 for v in candidates.values()):
        raise ValueError("Expected two ordinary and two recovery skill documents")
    skills = {f"{arm}_{i}": skill for arm, docs in candidates.items()
              for i, skill in enumerate(docs, 1)}
    if isinstance(evidence_ids, dict):
        evidence = {key: set(map(str, evidence_ids.get(key, ()))) for key in skills}
        shared = set().union(*evidence.values())
        for key in ("ordinary_1", "ordinary_2"):
            evidence[key] = shared
    else:
        evidence = {key: set(map(str, evidence_ids)) for key in skills}
    if any(used - set(ids) for used in evidence.values()):
        raise ValueError("Evidence contains an unknown dev ID")
    order = lineage_order(ids, history)
    anchor_score = sum(row["hard"] for row in reference.values()) / len(ids)
    manifest = {
        "anchor_sha256": _digest(anchor_skill), "anchor_correct": int(round(len(ids) * anchor_score)),
        "anchor_hard": [reference[key]["hard"] for key in ids],
        "anchor_execution_failures": [key for key in ids if not reference[key]["agent_ok"]],
        "candidate_sha256": {key: _digest(skill) for key, skill in skills.items()},
        "evidence_ids": {key: sorted(value) for key, value in evidence.items()},
        "dev_sha256": _digest(json.dumps(dev_items, ensure_ascii=False, sort_keys=True)),
        "order": order, "early_stop_batch": 4,
    }
    manifest_file = root / "evaluation_manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError(f"Resume manifest mismatch: {manifest_file}")
    _write(manifest_file, manifest)
    usage_path = root / "evaluation_usage.json"

    def screen(key):
        folder = root / "screen" / key
        same_as_anchor = skills[key].strip() == anchor_skill.strip()
        if same_as_anchor:
            # The original gate caches previously scored skills.
            result = {"status": "identical_to_anchor", "n": 200, "accuracy": anchor_score,
                      "lower_bound": anchor_score, "upper_bound": anchor_score,
                      "paired": _compare(list(reference.values()), reference, evidence[key])}
        else:
            outcome = evaluate_with_early_stop(
                items=dev_items, history=history, current_score=anchor_score, batch_size=4,
                run_items=lambda prefix: _answers(adapter, prefix, skills[key], folder),
            )
            score = None if outcome.stopped else outcome.lower_bound
            if outcome.stopped:
                status = "upper_bound_cannot_win"
            else:
                status = "strict_win" if score > anchor_score else "full_no_win"
            result = {"status": status, "n": outcome.n_evaluated, "accuracy": score,
                      "lower_bound": outcome.lower_bound, "upper_bound": outcome.upper_bound,
                      "paired": _compare(outcome.results, reference, evidence[key]),
                      "batches": outcome.batches}
        _write(root / f"screen_{key}.json", result)
        return key, result

    with _measure(usage_path, "screen"):
        with ThreadPoolExecutor(max_workers=min(4, max(1, adapter.workers // 4))) as pool:
            screening = dict(pool.map(screen, skills))
    winners = {}
    for arm in candidates:
        eligible = [key for key in skills if key.startswith(arm + "_")
                    and screening[key]["status"] == "strict_win"]
        if eligible:
            winners[arm] = max(eligible, key=lambda key: screening[key]["accuracy"])

    confirmation = {}
    if winners:
        with _measure(usage_path, "fresh_confirmation"):
            ordered_items = [dev_by_id[key] for key in order]
            fresh = _answers(adapter, ordered_items, anchor_skill, root / "confirmation" / "anchor")
            fresh_reference = _index(fresh)
            confirmation["anchor_accuracy"] = sum(row["hard"] for row in fresh) / len(fresh)
            for arm, key in winners.items():
                rows = _answers(adapter, ordered_items, skills[key], root / "confirmation" / key)
                comparison = _compare(rows, fresh_reference, evidence[key])
                confirmation[arm] = {"selected": key, "accuracy": sum(row["hard"] for row in rows) / len(rows),
                                     "beats_fresh_anchor": comparison["net_observed"] > 0, "paired": comparison}
    result = {
        "archived_anchor_accuracy": anchor_score, "screening": screening,
        "selected_for_confirmation": winners, "fresh_confirmation": confirmation,
        "usage": json.loads(usage_path.read_text()),
    }
    _write(root / "evaluation.json", result)
    return result
