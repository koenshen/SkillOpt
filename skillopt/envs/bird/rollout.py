"""BIRD rollout — single-turn text-to-SQL agent + batch execution.

The skill document is the system prompt and the item's pre-built ``prompt``
(task overview + schema + evidence + question + output format) is the user
message, as in the reference pipeline (``src/run_prompt.py``).  The final SQL
is extracted from the response and scored by execution accuracy.

Public API
----------
- :func:`process_one`  — run + evaluate one item
- :func:`run_batch`    — parallel, resume-aware execution of a list of items
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.bird.evaluator import DEFAULT_TIMEOUT, db_path, evaluate, preview_rows
from skillopt.model import chat_target, is_target_exec_backend


def _raise_on_systemic_failure(results: list[dict]) -> None:
    """Abort when all rollout rows failed before any agent response."""
    if not results or not all(row.get("agent_ok") is False for row in results):
        return
    reasons = Counter(str(row.get("fail_reason") or "unknown error") for row in results)
    common_reason, count = reasons.most_common(1)[0]
    raise RuntimeError(
        f"BIRD rollout failed for all {len(results)} items before an agent "
        f"response ({count}x): {common_reason}"
    )


def build_system(skill_content: str) -> str:
    return skill_content.strip()


def build_user(item: dict, *, diagnostic_mode: bool = False, diagnostic_instruction: str = "") -> str:
    user = str(item["prompt"])
    if diagnostic_mode and diagnostic_instruction.strip():
        user += f"\n\n## Training Readout\n{diagnostic_instruction.strip()}"
    return user


def _base_result(item: dict) -> dict:
    return {
        "id": str(item["id"]),
        "question_id": item.get("question_id"),
        "db_id": item.get("db_id", ""),
        "question": item.get("question", ""),
        "task_description": item.get("question", ""),
        "task_type": item.get("difficulty") or item.get("task_type") or "bird",
        "hard": 0,
        "soft": 0.0,
        "predicted_sql": "",
        "gold_sql": item.get("SQL", ""),
        "exec_status": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }


def _eval_detail(item: dict, ev: dict) -> str:
    lines = [
        "[EVALUATION RESULT]",
        f"Database: {item.get('db_id', '')}",
        f"Question: {item.get('question', '')}",
        f"Evidence: {item.get('evidence', '')}",
        f"Predicted SQL: {ev['pred_sql'] or '(none extracted)'}",
        f"Gold SQL: {item.get('SQL', '')}",
        f"Execution: {ev['status']}" + (f" — {ev['error']}" if ev["error"] else ""),
        f"Predicted rows: {preview_rows(ev['pred_rows'])}",
        f"Gold rows: {preview_rows(ev['gold_rows'])}",
        f"Execution accuracy (set of rows equal): {ev['res']}",
    ]
    return "\n".join(lines)


def _fail_reason(ev: dict) -> str:
    if ev["res"]:
        return ""
    if not ev["pred_sql"]:
        return "EX=0: no SQL extracted from the response"
    if ev["status"] == "timeout":
        return f"EX=0: {ev['error']}"
    if ev["status"] == "error":
        return f"EX=0: execution error: {ev['error'][:300]}"
    return (
        f"EX=0: result mismatch (pred {len(ev['pred_rows'] or [])} rows, "
        f"gold {len(ev['gold_rows'] or [])} rows)"
    )


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    db_root: str,
    *,
    exec_timeout: int = 300,
    sql_timeout: float = DEFAULT_TIMEOUT,
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> dict:
    """Run one item: one chat call, extract SQL, execute and score."""
    result = _base_result(item)
    try:
        pred_dir = os.path.join(out_root, "predictions", result["id"])
        os.makedirs(pred_dir, exist_ok=True)

        system = build_system(skill_content)
        user = build_user(item, diagnostic_mode=diagnostic_mode, diagnostic_instruction=diagnostic_instruction)
        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=exec_timeout,
        )
        response = response or ""
        result.update(response=response, agent_ok=True, n_turns=1)
        conversation: list[dict] = [{"type": "message", "turn": 1, "content": response}]

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)

        ev = evaluate(response, str(item["SQL"]), db_path(db_root, item["db_id"]), timeout=sql_timeout)
        result.update(
            hard=int(ev["res"]),
            soft=float(ev["res"]),
            predicted_sql=ev["pred_sql"],
            exec_status=ev["status"],
            fail_reason=_fail_reason(ev),
        )
        conversation.append({"role": "system", "content": _eval_detail(item, ev)})
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    db_root: str,
    *,
    exec_timeout: int = 300,
    sql_timeout: float = DEFAULT_TIMEOUT,
    workers: int = 24,
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    task_timeout: int = 0,
) -> list[dict]:
    """Run all items with a thread pool. Resume-aware via ``results.jsonl``."""
    if is_target_exec_backend():
        raise NotImplementedError("BIRD env supports chat target backends only")
    if not os.path.isdir(db_root):
        raise FileNotFoundError(f"BIRD db_root not found: {db_root}")
    task_timeout = max(int(task_timeout), int(exec_timeout) + int(sql_timeout) + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:  # noqa: BLE001
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        _raise_on_systemic_failure(existing)
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)
    results = list(existing)

    def _timeout_result(item: dict) -> dict:
        row = _base_result(item)
        row.update(fail_reason=f"task-timeout-{task_timeout}s", phase="timeout")
        return row

    def _error_result(item: dict, exc: Exception) -> dict:
        row = _base_result(item)
        row.update(fail_reason=f"unexpected: {type(exc).__name__}: {exc}", phase="error")
        return row

    started_at: dict[str, float] = {}

    def _run_one(item: dict) -> dict:
        started_at[str(item["id"])] = time.time()
        return process_one(
            item,
            out_root,
            skill_content,
            db_root,
            exec_timeout=exec_timeout,
            sql_timeout=sql_timeout,
            max_completion_tokens=max_completion_tokens,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
        )

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    item = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _error_result(item, exc)
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct_count += 1
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"(acc={correct_count / completed:.3f}) id={res['id']} "
                        f"hard={res.get('hard', '?')} {res.get('exec_status', '')}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    fut.cancel()
                    res = _timeout_result(futs[fut])
                    results.append(res)
                    completed += 1
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"(acc={correct_count / completed:.3f}) id={res['id']} TIMEOUT",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    _raise_on_systemic_failure(results)
    return results
