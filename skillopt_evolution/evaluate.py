"""Screen a child against its own parent on a generation's frozen dev order.

Call slots sequentially when using the process-wide provider token tracker. Items
within a paid batch may run concurrently in the original benchmark adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .usage import measure_usage, read_json, save_json


class _AdapterFailure(Exception):
    """Distinguish rollout errors from I/O or accounting failures."""


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _check_row(key, row):
    if (str(row["id"]) != key or row["hard"] not in (0, 1)
            or not isinstance(row["agent_ok"], bool)):
        raise ValueError(f"Invalid dev score for {key!r}")


def _read_attempt(directory, requested):
    """Recover even a partially written adapter sheet after an interruption."""
    saved = directory / "answers.json"
    if saved.exists():
        rows = read_json(saved)
    else:
        raw = directory / "rollout" / "results.jsonl"
        rows = []
        if raw.exists():
            with raw.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            if line.endswith("\n"):
                                raise
                            # An unfinished final line provides no valid observation.
    seen = set()
    for row in rows:
        key = str(row["id"])
        _check_row(key, row)
        if key not in requested or key in seen:
            raise ValueError(f"Unexpected or duplicate answer {key!r} in {directory}")
        seen.add(key)
    return rows


def evaluate_candidate(*, adapter, dev_items, order, skill, parent_rows,
                       cached_rows, out_dir, batch_size=4, cache_sources=None, require_complete=False):
    """Use all previously observed IDs to bound the full dev count at each batch.

    Cache keys outside this function are exact skill-text hashes in a single
    source run. The runner passes a snapshot and incorporates returned rows.
    Returning `accuracy=None` on a partial sheet prevents ranking by a bound.
    """
    directory = Path(out_dir)
    order = [str(key) for key in order]
    items = {str(item["id"]): item for item in dev_items}
    n = len(order)
    if (not n or n != len(dev_items) or len(items) != n or set(order) != set(items)
            or set(parent_rows) != set(items) or not set(cached_rows).issubset(items)):
        raise ValueError("Frozen order, complete parent and cached rows must use the same dev IDs")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for key, row in parent_rows.items():
        _check_row(key, row)
    for key, row in cached_rows.items():
        _check_row(key, row)
    parent_correct = int(sum(row["hard"] for row in parent_rows.values()))
    signature = {
        "child_sha256": hashlib.sha256(skill.encode()).hexdigest(),
        "parent_rows_sha256": _digest(parent_rows), "parent_correct": parent_correct,
        "dev_sha256": _digest(dev_items), "frozen_order": order, "batch_size": batch_size,
    }
    if require_complete:
        signature["require_complete"] = True
    manifest = directory / "evaluation_inputs.json"
    if manifest.exists():
        initial = read_json(manifest)
        if initial["signature"] != signature:
            raise ValueError(f"Resume inputs changed in {directory}")
    else:
        # Freeze cache provenance before any calls, including rows from earlier
        # slots of this generation. Subsequent resumes use this same snapshot.
        initial = {"signature": signature, "cached_rows": cached_rows, "cache_sources": cache_sources or {}}
        save_json(manifest, initial)

    result_file = directory / "evaluation.json"
    if result_file.exists():
        return read_json(result_file)
    cached = initial["cached_rows"]
    observed = {key: row for key, row in cached.items() if row["agent_ok"]}
    reused = set(observed)
    sources = {k: initial["cache_sources"].get(k, "input_cache") for k in observed}
    prior_failed = [k for k in order if k in cached and not cached[k]["agent_ok"]]

    def absorb(rows, requested, source):
        failed = []
        received = set()
        for row in rows:
            k = str(row["id"])
            _check_row(k, row)
            if k not in requested or k in received:
                raise ValueError(f"Unexpected or duplicate dev result {k!r} from {source}")
            received.add(k)
            if row["agent_ok"]:
                if k in observed and row != observed[k]:
                    raise ValueError(f"Conflicting cached answer for {k!r}")
                observed[k] = row
                sources[k] = source
            else:
                failed.append(k)
        return failed + [k for k in requested if k not in received]

    attempts = directory / "evaluation_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempts.glob("*/request.json"))
    for request in previous:
        wanted = read_json(request)["ids"]
        restored = _read_attempt(request.parent, wanted)
        absorb(restored, wanted, str(request.parent.relative_to(directory)))
    serial = max((int(p.parent.name) for p in previous), default=0)
    batches = []
    while True:
        correct = int(sum(row["hard"] for row in observed.values()))
        upper = correct + n - len(observed)
        batches.append({"observed": len(observed), "correct": correct, "upper_count": upper})
        if len(observed) == n or (not require_complete and upper <= parent_correct):
            break
        # Old cached answers may be outside this generation's prefix. They
        # already constrain the bound; only unanswered IDs cost more tokens.
        missing = [k for k in order if k not in observed][:batch_size]
        for attempt in (1, 2):
            serial += 1
            path = attempts / f"{serial:05d}"
            ids = list(missing)
            error = None
            try:
                with measure_usage(directory, "validation") as token_event:
                    save_json(path / "request.json", {
                        "ids": ids, "attempt_in_batch": attempt,
                        "prior_observed": len(observed), "upper_count_before_batch": upper,
                        "token_event": str(token_event.relative_to(directory.resolve())),
                    })
                    try:
                        rows = adapter.rollout([items[k] for k in ids], skill, str(path / "rollout"))
                    except Exception as exc:
                        raise _AdapterFailure(type(exc).__name__) from exc
                    save_json(path / "answers.json", rows)
            except _AdapterFailure as exc:
                # Some adapters raise if every item failed; retain all evidence
                # written before the exception and retry only unresolved items.
                error = str(exc)
                rows = _read_attempt(path, ids)
            missing = absorb(rows, ids, str(path.relative_to(directory)))
            measured_correct = int(sum(row["hard"] for row in observed.values()))
            save_json(path / "outcome.json", {
                "observed": len(observed), "correct": measured_correct,
                "upper_count": measured_correct + n - len(observed), "parent_correct": parent_correct,
                "successful_ids": [k for k in ids if k not in missing],
                "failed_or_missing_ids": missing, "adapter_exception": error,
            })
            if not missing:
                break
        if missing:
            raise RuntimeError(f"Unresolved dev requests after retry in {path}: {missing!r}")

    complete = len(observed) == n
    ordered = [k for k in order if k in observed]
    new_ids = [k for k in ordered if k not in reused]
    fixes = [k for k in ordered if parent_rows[k]["agent_ok"] and observed[k]["hard"] > parent_rows[k]["hard"]]
    regressions = [k for k in ordered if parent_rows[k]["agent_ok"] and observed[k]["hard"] < parent_rows[k]["hard"]]
    result = {
        "accepted": complete and correct > parent_correct, "complete": complete,
        "decision": ("strict_win" if correct > parent_correct else "full_no_win") if complete else "bound_reject",
        "accuracy": correct / n if complete else None, "correct": correct, "n_observed": len(observed),
        "n_total": n, "parent_correct": parent_correct,
        "lower_bound": correct / n, "upper_bound": upper / n,
        "fixes": fixes, "regressions": regressions,
        "parent_failed_observed_ids": [k for k in ordered if not parent_rows[k]["agent_ok"]],
        "new_ids": new_ids, "reused_ids": [k for k in ordered if k in reused],
        "prior_failed_ids": prior_failed, "recovered_failure_ids": [k for k in prior_failed if k in observed],
        "row_sources": sources, "rows": observed, "bound_trace": batches,
    }
    save_json(result_file, result)
    return result
