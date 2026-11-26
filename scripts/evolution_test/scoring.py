"""Score a frozen skill on the full test split, preserving completed answers on resume."""
from __future__ import annotations

import json
import math
from pathlib import Path

from skillopt.model import get_token_summary
from skillopt_evolution.usage import now, read_json, save_json, zero_usage


def _rows(path):
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise  # An unfinished trailing row may be retried.


def _absorb(latest, rows, requested):
    seen = set()
    for row in rows:
        key = str(row["id"])
        if key not in requested or key in seen or type(row["agent_ok"]) is not bool:
            raise ValueError(f"Unexpected, invalid, or duplicate test result for {key}")
        if row["agent_ok"]:
            if row["hard"] not in (0, 1) or not 0 <= row["soft"] <= 1:
                raise ValueError(f"Invalid test EM/F1 for {key}")
        if latest.get(key, {}).get("agent_ok") and latest[key] != row:
            raise ValueError(f"Conflicting successful answers for {key}")
        latest[key] = row
        seen.add(key)


def _usage(attempts):
    result = zero_usage()
    result.update(unfinished_attempts=[], zero_reported_attempts=[])
    for path in sorted(attempts.glob("*/request.json")):
        measured = path.parent / "usage.json"
        if not measured.is_file():
            result["unfinished_attempts"].append(path.parent.name)
            continue
        stats = read_json(measured)
        if not stats["total_tokens"]:
            result["zero_reported_attempts"].append(path.parent.name)
        for k in ("calls", "input_tokens", "output_tokens", "total_tokens"):
            result[k] += stats[k]
    return result


def run_test(*, adapter, items, skill, out_dir):
    """The caller owns the run lock and pins model, split and skill in its manifest."""
    root = Path(out_dir)
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    by_id = {str(item["id"]): item for item in items}
    if not items or len(by_id) != len(items):
        raise ValueError("Test items must have distinct IDs")
    workers = int(adapter.workers)
    if workers < 1:
        raise ValueError("Test worker count must be positive")
    latest = {}
    saved = sorted(attempts.glob("*/request.json"))
    for path in saved:
        requested = set(read_json(path)["ids"])
        if not requested.issubset(by_id):
            raise ValueError(f"Stored items outside the frozen test split in {path}")
        _absorb(latest, _rows(path.parent / "results.jsonl"), requested)
    serial = max((int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdecimal()), default=0)
    pending = [k for k in by_id if not latest.get(k, {}).get("agent_ok")]

    for start in range(0, len(pending), workers):
        batch = pending[start:start + workers]
        missing = list(batch)
        for attempt in (1, 2):
            serial += 1
            folder = attempts / f"{serial:05d}"
            folder.mkdir()
            save_json(folder / "request.json", {"ids": missing, "attempt_in_batch": attempt, "started_utc": now()})
            before = get_token_summary()["_total"].copy()
            try:
                rows = adapter.rollout([by_id[k] for k in missing], skill, str(folder))
            except Exception as error:
                save_json(folder / "error.json", {"exception": type(error).__name__, "time": now()})
                # Native QA writes each answer as soon as it returns, even
                # when the final all-failed check raises. Keep those records.
                rows = list(_rows(folder / "results.jsonl"))
            finally:
                after = get_token_summary()["_total"]
                save_json(folder / "usage.json", {
                    dst: after[src] - before[src] for dst, src in (
                        ("calls", "calls"), ("input_tokens", "prompt_tokens"),
                        ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens"),
                    )
                })
            _absorb(latest, rows, set(missing))
            missing = [k for k in missing if not latest.get(k, {}).get("agent_ok")]
            save_json(folder / "outcome.json", {"failed_or_missing_ids": missing, "finished_utc": now()})
            if not missing:
                break
        costs = _usage(attempts)
        save_json(root / "usage.json", costs)
        n_ok = sum(row["agent_ok"] for row in latest.values())
        save_json(root / "progress.json", {"n_total": len(items), "n_answered": n_ok,
                                           "latest_attempt": serial, "updated_utc": now()})
        print(f"Test successful answers {n_ok}/{len(items)}, latest attempt {serial}", flush=True)
        if len(missing) == workers and len(batch) == workers:
            raise RuntimeError(f"Whole test batch failed twice; inspect {folder} and resume")

    ordered = [latest.get(k, {"id": k, "question": by_id[k].get("question"), "agent_ok": False,
                              "hard": 0, "soft": 0.0, "fail_reason": "No result after retry"}) for k in by_id]
    failed = [str(row["id"]) for row in ordered if not row["agent_ok"]]
    correct = sum(row["hard"] for row in ordered if row["agent_ok"])
    tmp = root / "results.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(root / "results.jsonl")
    costs = _usage(attempts)
    save_json(root / "usage.json", costs)
    return {"status": "failed_items" if failed else "complete",
            "n_total": len(items), "n_correct": correct, "n_answered": len(items)-len(failed),
            "n_failed": len(failed), "failed_ids": failed,
            "em": correct / len(items),
            "f1": math.fsum(row["soft"] for row in ordered if row["agent_ok"]) / len(items),
            "usage": costs}
