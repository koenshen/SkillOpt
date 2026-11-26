"""Read one SkillOpt run's dev evidence without loading any held-out split."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from skillopt.evaluation.early_stop import lineage_order, update_history
from skillopt.prompts import load_prompt


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _system_for(text, env):
    if env == "bird":
        return text.strip()
    if env != "searchqa":
        raise ValueError(f"Unsupported archived dev prompt format for {env!r}")
    section = f"## Skill\n{text.strip()}\n\n" if text.strip() else ""
    return load_prompt("rollout_system", env="searchqa").format(skill_section=section)


def _system_path(folder):
    return next((folder / "predictions").glob("*/target_system_prompt.txt"), None)


def _sheet(source, sheet, text_file, uid, step, action, universe, env, parent=None, edits=None):
    text = text_file.read_text(encoding="utf-8")
    expected_prompt = _system_for(text, env).encode("utf-8")
    rows = {}
    with (sheet / "results.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            if key in rows or key not in universe:
                raise ValueError(f"Duplicate or unexpected dev ID {key!r} in {sheet}")
            if row["hard"] not in (0, 1) or not isinstance(row["agent_ok"], bool):
                raise ValueError(f"Expected a binary dev result and agent_ok for {key} in {sheet}")
            if row["agent_ok"]:
                prompt = sheet / "predictions" / key / "target_system_prompt.txt"
                if not prompt.is_file() or prompt.read_bytes() != expected_prompt:
                    raise ValueError(f"Cannot match {text_file} to archived system prompt {prompt}")
            rows[key] = {k: v for k, v in row.items() if k in (
                "id", "hard", "agent_ok", "question", "predicted_answer", "gold_answers",
                "predicted_sql", "gold_sql", "db_id", "exec_status", "f1",
            )}
    if not rows or not any(row["agent_ok"] for row in rows.values()):
        raise ValueError(f"No successful dev observation in {sheet}")
    full = rows.keys() == universe
    correct = int(sum(r["hard"] for r in rows.values())) if full else None
    return {
        "id": uid, "step": step, "action": action, "sha256": _sha(text), "skill": text,
        "source": str(text_file.relative_to(source)),
        "sheet": str((sheet / "results.jsonl").relative_to(source)),
        "parent_sha256": parent, "edits": edits or [], "rows": rows,
        "complete": full, "correct": correct, "score": correct / len(rows) if full else None,
    }


def _check_prefix(rec, cut, ids, history):
    order = lineage_order(ids, history)
    rows = rec["rows"]
    n = len(rows)
    if (cut["n_evaluated"] != n or cut["n_total"] != len(ids)
            or set(order[:n]) != rows.keys() or cut["stopped"] != (n < len(ids))):
        raise ValueError(f"Invalid historical dev prefix at {rec['id']}")
    size = cut["batch_size"]
    if size <= 0:
        raise ValueError(f"Invalid historical batch size at {rec['id']}")
    expected = list(range(size, n, size)) + [n]
    if expected != [batch["n_evaluated"] for batch in cut["batches"]]:
        raise ValueError(f"Invalid batch boundaries at {rec['id']}")
    for end, batch in zip(expected, cut["batches"]):
        if sum(rows[k]["hard"] for k in order[:end]) != batch["hard_sum"]:
            raise ValueError(f"Invalid cumulative dev count at {rec['id']}, batch {end}")


def _applied(folder):
    report = folder / "edit_apply_report.json"
    if not report.exists():
        return []
    applied = [edit for edit in _json(report) if edit["status"].startswith("applied")]
    if not applied:
        return []
    ranked = _json(folder / "ranked_edits.json")["edits"]
    result = []
    for entry in applied:
        index = entry["index"]
        if not 1 <= index <= len(ranked) or ranked[index - 1]["op"] != entry["op"]:
            raise ValueError(f"Invalid source edit index in {report}")
        edit = ranked[index - 1]
        result.append(dict(index=index, op=entry["op"], target=edit.get("target", ""),
                           content=edit.get("content", ""), status=entry["status"]))
    return result


def _initial_population(records, archive_best_sha, cap):
    full = [r for r in records if r["complete"]]
    # A stochastic re-evaluation is another observation, not another parent.
    # Prefer the highest fully measured score; ties use the archived best and
    # then the deterministic first record, never the step's early-stop bound.
    ranked = sorted(full, key=lambda r: (-r["correct"], r["sha256"] != archive_best_sha))
    unique = {}
    for r in ranked:
        unique.setdefault(r["sha256"], r)
    population = list(unique.values())[:cap]
    chosen = {r["id"] for r in population}
    inventory = []
    for r in records:
        other = unique.get(r["sha256"])
        if not r["complete"]:
            reason = "partial_dev"
        elif r["id"] in chosen:
            reason = "selected"
        elif r is not other:
            reason = "duplicate_text"
        else:
            reason = "population_cap"
        entry = {k: r[k] for k in ("id", "step", "action", "source", "sheet", "sha256", "correct", "score")}
        entry.update(reason=reason, selected=r["id"] in chosen, dev_observed=len(r["rows"]))
        if reason == "duplicate_text":
            entry["canonical_id"] = other["id"]
        inventory.append(entry)
    return population, {
        "population_cap": cap, "complete_sheets": len(full),
        "unique_complete_skills": len(unique), "selected_count": len(population),
        "shortfall": cap - len(population), "excluded_by_cap": max(0, len(unique) - cap),
        "duplicate_complete_sheets": len(full) - len(unique),
        "tie_break": "highest full score; archived best in score ties; then first recorded",
        "selected": [r["id"] for r in population], "records": inventory,
    }


def load_archive(source: Path, dev_ids: list[str], population_cap: int = 5) -> dict:
    """Replay the native difficulty history; seed the separate evolution population.

    SkillOpt adds baseline and ordinary selection results to its ordering
    history. Measured final/slow validation sheets are eligible parents but
    were never added to that ordering, so do not replay them as ordinary steps.
    """
    if not 1 <= population_cap <= 5:
        raise ValueError("population_cap must be between 1 and 5")
    source = Path(source)
    ids = list(map(str, dev_ids))
    universe = set(ids)
    if not ids or len(universe) != len(ids):
        raise ValueError("Expected unique nonempty dev IDs")
    cfg = _json(source / "config.json")
    env = cfg["env"]
    history, skipped = {}, []
    native = _json(source / "history.json")
    steps = [r["step"] for r in native]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(f"Incomplete native step history in {source}")
    last = steps[-1] if steps else 0
    final_dir = source / "final_selection_eval"
    if not ((final_dir / "results.jsonl").is_file() or (final_dir / "selection.json").is_file()):
        # Native runs that disable held-out evaluation omit the final dev sheet.
        # Check the terminal marker's existence only; never read its test scores.
        runtime_path = source / "runtime_state.json"
        runtime = _json(runtime_path) if runtime_path.is_file() else {}
        if (not (source / "summary.json").is_file() or runtime.get("last_completed_step") != last
                or cfg.get("eval_test", True)):
            raise ValueError(f"No completed final dev record or completed dev-only run at {source}")
    snapshots = source / "skills"
    baseline = _sheet(source, source / "selection_eval_baseline", snapshots / "skill_v0000.md",
                      "v0", 0, "initial", universe, env)
    if not baseline["complete"]:
        raise ValueError("Missing initial dev items")
    records = [baseline]
    update_history(history, list(baseline["rows"].values()))
    tokens = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    for item in native:
        step = item["step"]
        uid = f"step_{step:04d}"
        folder = source / "steps" / uid
        scores = folder / "selection_eval"
        cut = item.get("selection_early_stop")
        for stage, cost in item.get("tokens", {}).items():
            if stage != "_total":
                tokens["input_tokens"] += cost["prompt_tokens"]
                tokens["output_tokens"] += cost["completion_tokens"]
                tokens["calls"] += cost.get("calls", 0)
        if not (scores / "results.jsonl").is_file():
            if cut:
                raise ValueError(f"Recorded a fresh evaluation but no answer sheet at {folder}")
            skipped.append({"id": uid, "step": step, "action": item["action"],
                            "reason": "no_new_dev_observations", "candidate_hash": item.get("candidate_hash")})
            continue
        parent_text = (snapshots / f"skill_v{step - 1:04d}.md").read_text()
        parent_prompt = _system_path(folder / "rollout")
        if parent_prompt is None:  # Native accumulation uses one rollout directory per train batch.
            parent_prompt = next(folder.glob("batch_*/rollout/predictions/*/target_system_prompt.txt"), None)
        # Epoch updates and skill-aware appendices can change the effective parent.
        # If an archive cannot verify this exact text, leave ancestry unknown.
        parent_hash = None
        if parent_prompt is not None and parent_prompt.read_text() == _system_for(parent_text, env):
            parent_hash = _sha(parent_text)
        rec = _sheet(source, scores, folder / "candidate_skill.md", uid, step, item["action"],
                     universe, env, parent_hash, _applied(folder))
        if item.get("candidate_hash") and rec["sha256"][:16] != item["candidate_hash"]:
            raise ValueError(f"Candidate hash mismatch at {uid}")
        if cut:
            _check_prefix(rec, cut, ids, history)
        elif not rec["complete"]:
            raise ValueError(f"Missing early-stop metadata for partial sheet {uid}")
        if rec["complete"] and not math.isclose(rec["score"], item["selection_hard"]):
            raise ValueError(f"Full answer sheet disagrees with selection_hard at {uid}")
        records.append(rec)
        update_history(history, list(rec["rows"].values()))

    for sheet in sorted((source / "slow_update").glob("epoch_*/selection_eval/results.jsonl")):
        folder = sheet.parent.parent
        slow = _json(folder / "slow_result.json")
        rec = _sheet(source, sheet.parent, folder / "candidate_skill.md", f"slow_{folder.name}",
                     None, slow["action"], universe, env)
        if not rec["complete"]:
            raise ValueError(f"Incomplete slow-update dev evaluation at {folder}")
        records.append(rec)

    final_file = snapshots / f"skill_v{last:04d}.md"
    best_file = source / "best_skill.md"
    best_sha = _sha(best_file.read_text()) if best_file.is_file() else None
    if (final_dir / "results.jsonl").is_file():
        final = _sheet(source, final_dir, final_file, "final", last, "final", universe, env)
        if not final["complete"]:
            raise ValueError("Incomplete final validation in the source run")
        records.append(final)
    elif (final_dir / "selection.json").exists():
        marker = _json(final_dir / "selection.json")
        if marker.get("reused_from") != "best" or _sha(final_file.read_text()) != best_sha:
            raise ValueError("Final dev reuse marker does not match archived best")
        skipped.append({"id": "final", "reason": "reused_best_sheet", "sha256": best_sha})
    if best_sha is not None and not any(r["complete"] and r["sha256"] == best_sha for r in records):
        raise ValueError("Cannot associate the native best with a complete dev answer sheet")

    population, initial = _initial_population(records, best_sha, population_cap)
    return {"source_root": str(source.resolve()), "records": records, "population": population, "history": history,
            "initial_population": initial, "skipped": skipped,
            "native_step_usage": {**tokens, "total_tokens": tokens["input_tokens"] + tokens["output_tokens"],
                                  "scope": "history.json per-step token deltas; excludes costs outside native steps"}}
