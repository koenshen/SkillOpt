"""Recover donor evidence from completed QA dev evaluations in one SkillOpt run."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from skillopt.evaluation.early_stop import lineage_order, update_history


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(folder: Path) -> dict:
    with (folder / "results.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f"Duplicate question ID in {folder}")
    return by_id


def _system_hash(folder: Path) -> str:
    # QA puts the skill in the same system message for the whole rollout;
    # question/evidence are in the per-item user message.
    path = next((folder / "predictions").glob("*/target_system_prompt.txt"), None)
    if path is None:
        raise ValueError(f"Missing archived system prompt in {folder}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _applied_edits(step_dir: Path) -> list[dict]:
    ranked = _json(step_dir / "ranked_edits.json")["edits"]
    result = []
    for record in _json(step_dir / "edit_apply_report.json"):
        if record["status"].startswith("applied"):
            index = record["index"]
            edit = ranked[index - 1]
            if index < 1 or edit["op"] != record["op"]:
                raise ValueError(f"Edit index disagrees with application report: {step_dir}")
            result.append({
                "index": index, "op": edit["op"], "target": edit.get("target", ""),
                "content": edit.get("content", ""), "application": record["status"],
            })
    return result


@dataclass
class _Donor:
    step: int
    parent_step: int
    parent: dict
    rows: dict
    edits: list
    gain: set
    loss: set


def _make_bank(donors: list[_Donor], anchor: dict, best_step: int) -> tuple[dict, list[str]]:
    # Build sibling success denominators only on items actually attempted and
    # successfully executed. Unobserved answers never enter a comparison.
    siblings = defaultdict(list)
    for donor in donors:
        siblings[donor.parent_step].append(donor)

    def sibling_counts(donor, item):
        others = [
            peer.rows[item]["hard"] for peer in siblings[donor.parent_step]
            if peer.step != donor.step and item in peer.rows and peer.rows[item]["agent_ok"]
        ]
        return int(sum(others)), len(others)

    def joint_gains(donor):
        return {item for item in donor.gain if anchor[item]["agent_ok"] and not anchor[item]["hard"]}

    joint = {d.step: joint_gains(d) for d in donors}
    repeat = {
        d.step: {k for k in joint[d.step] if sibling_counts(d, k)[0] > 0}
        for d in donors
    }
    primary = []
    seen_items = set()
    remaining = [d for d in donors if joint[d.step] and d.edits]
    # Prioritize distinct anchor weaknesses, then repairs also seen in siblings.
    while remaining and len(primary) < 5:
        donor = max(remaining, key=lambda d: (
            len(joint[d.step] - seen_items), len(repeat[d.step]), len(joint[d.step]),
            -len(d.loss), -d.step,
        ))
        primary.append(donor)
        remaining.remove(donor)
        seen_items.update(joint[donor.step])

    # Include close contrasts, so the model can inspect what the other children
    # actually changed on the same parent, even if those children won no points.
    comparisons = []
    selected_steps = {d.step for d in primary}
    for donor in primary:
        available = [
            d for d in siblings[donor.parent_step] if d.step not in selected_steps and d.edits
        ]
        if available:
            def contrast(peer):
                overlap = [k for k in joint[donor.step] if k in peer.rows and peer.rows[k]["agent_ok"]]
                return sum(not peer.rows[k]["hard"] for k in overlap), len(overlap), -peer.step

            peer = max(available, key=contrast)
            if contrast(peer)[1]:
                comparisons.append((peer, donor))
                selected_steps.add(peer.step)

    bank = {
        "anchor_step": best_step,
        "scope": "Rejected actual edits, same-run exact-parent matches and observed dev answers only.",
        "legend": (
            "Each example shows answers and EM for anchor, actual parent and rejected child. "
            "sibling_em is [number correct, number observed] among OTHER rejected children "
            "with this exact parent prompt. Counts refer to a biased, early-stopped prefix. "
            "Sibling agreement alone does not identify an edit in a multi-edit child. "
            "Use these observations to hypothesize a general rule; never reproduce individual "
            "dev questions, answers, named examples or lookup exceptions in a skill."
        ),
        "donors": [],
    }
    evidence_ids = set()
    requests = [(d, None) for d in primary] + comparisons
    for donor, compared_to in requests:
        evidence_items = joint[donor.step] if compared_to is None else joint[compared_to.step]
        evidence_items = {k for k in evidence_items if k in donor.rows and donor.rows[k]["agent_ok"]}
        positives = sorted(evidence_items, key=lambda k: (k not in repeat[donor.step], k))[:5]
        risk = sorted(donor.loss, key=lambda k: (-int(anchor[k]["hard"]), k))[:5]
        ids = list(dict.fromkeys(positives + risk))

        def answer(row):
            return {"response": str(row.get("predicted_answer", ""))[:200], "em": int(row["hard"])}

        examples = [{
            "id": k, "question": str(anchor[k]["question"])[:360], "anchor": answer(anchor[k]),
            "parent": answer(donor.parent[k]), "child": answer(donor.rows[k]),
            "sibling_em": sibling_counts(donor, k),
        } for k in ids]
        item = {
            "step": donor.step, "parent_step": donor.parent_step,
            "comparison_for_step": compared_to.step if compared_to else None,
            "evaluated": len(donor.rows), "parent_repairs": len(donor.gain),
            "parent_regressions": len(donor.loss), "also_repairs_anchor": len(joint[donor.step]),
            "edits": donor.edits, "examples": examples,
        }
        bank["donors"].append(item)
        if len(json.dumps(bank, ensure_ascii=False)) > 70000:
            bank["donors"].pop()
            continue
        evidence_ids.update(ids)
    if not bank["donors"]:
        raise ValueError("No supported donor fits the evidence budget")
    return bank, sorted(evidence_ids)


def build_dev_archive(run_dir: Path, dev_ids: list[str]) -> dict:
    """Replay completed selection rounds and construct an automatic evidence bank."""
    run_dir = Path(run_dir)
    ids = [str(k) for k in dev_ids]
    universe = set(ids)
    records = _json(run_dir / "history.json")
    best_step = int(records[-1]["best_step"])
    best_skill = (run_dir / "best_skill.md").read_text(encoding="utf-8")
    best_dir = run_dir / "steps" / f"step_{best_step:04d}"
    if best_skill != (best_dir / "candidate_skill.md").read_text(encoding="utf-8"):
        raise ValueError("Archived best skill does not match the candidate evaluated on dev")
    anchor = _rows(best_dir / "selection_eval")
    initial_dir = run_dir / "selection_eval_baseline"
    initial = _rows(initial_dir)
    if len(ids) != len(universe) or set(anchor) != universe or set(initial) != universe:
        raise ValueError("Initial/best dev coverage differs from the configured split")
    best_hard = sum(r["hard"] for r in anchor.values()) / len(ids)
    if not math.isclose(best_hard, records[-1]["best_score"]):
        raise ValueError("Per-item best score differs from recorded best dev score")

    history = {}
    update_history(history, list(initial.values()))
    complete = {_system_hash(initial_dir): (0, initial)}
    donors = []
    skipped_parents = []
    for rec in records:
        step = int(rec["step"])
        folder = run_dir / "steps" / f"step_{step:04d}"
        eval_dir = folder / "selection_eval"
        rows = _rows(eval_dir)
        order = lineage_order(ids, history)
        cut = rec["selection_early_stop"]
        if cut["n_evaluated"] != len(rows) or set(order[:len(rows)]) != set(rows):
            raise ValueError(f"Dev prefix differs from historical ranking at step {step}")
        for batch in cut["batches"]:
            done = batch["n_evaluated"]
            if sum(rows[k]["hard"] for k in order[:done]) != batch["hard_sum"]:
                raise ValueError(f"Archived batch score mismatch at step {step}")
        if rec["action"] == "reject":
            parent_match = complete.get(_system_hash(folder / "rollout"))
            if parent_match is None:
                skipped_parents.append(step)
            else:
                parent_step, parent = parent_match
                valid = [k for k in rows if rows[k]["agent_ok"] and parent[k]["agent_ok"]
                         and anchor[k]["agent_ok"]]
                gain = {k for k in valid if rows[k]["hard"] > parent[k]["hard"]}
                loss = {k for k in valid if rows[k]["hard"] < parent[k]["hard"]}
                donors.append(_Donor(step, parent_step, parent, rows, _applied_edits(folder), gain, loss))
        if len(rows) == len(ids):
            complete[_system_hash(eval_dir)] = (step, rows)
        update_history(history, list(rows.values()))
    bank, evidence_ids = _make_bank(donors, anchor, best_step)
    return {
        "best_step": best_step, "best_skill": best_skill, "best_hard": best_hard,
        "best_rows": [anchor[k] for k in ids], "final_dev_order": lineage_order(ids, history),
        "history": history, "evidence": bank, "evidence_ids": evidence_ids,
        "audit": {"rounds": len(records), "exact_parent_rejects": len(donors),
                  "unmatched_parent_steps": skipped_parents},
    }
