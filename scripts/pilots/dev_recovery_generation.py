"""Candidate generation for the standalone, dev-only recovery experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from skillopt.engine.trainer import _normalise_patches
from skillopt.gradient.aggregate import merge_patches
from skillopt.model import chat_optimizer, get_token_summary
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.utils import extract_json


def _write(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _save(path: Path, contents) -> None:
    _write(path, json.dumps(contents, indent=2, ensure_ascii=False))


def _checkpoint(path: Path, run):
    """Persist each completed stage and its token usage together."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    before = get_token_summary().get("_total", {}).copy()
    result = run()
    after = get_token_summary().get("_total", {})
    usage = {k: after.get(k, 0) - before.get(k, 0)
             for k in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}
    record = {"tokens": usage, "result": result}
    _save(path, record)
    return record


def _recovery_patch(anchor, bank, merged, meta, budget, directory):
    system = (
        "You are the optimizer in a controlled skill-evolution experiment. "
        "Choose historical edits yourself. Look for a reusable change that multiple "
        "rejected siblings made when they repaired an anchor weakness. Inspect other "
        "edits in those siblings, sibling controls, regressions, and partial coverage. "
        "A single flipped answer is weak evidence; a skipped item provides no outcome. "
        "Decide on a minimal change that generalizes beyond the observed items. "
        "Use the current training feedback to adapt it to this anchor. Define when "
        "the rule should apply and when existing behavior should be preserved. "
        "Do not copy dev questions, answer literals, IDs or entity-specific exceptions "
        "into the skill. The resulting skill must be usable on fresh tasks. "
        "Return JSON: {decision: 'edit' or 'abstain', selected: [{step: integer, "
        "index: integer}], rationale: string, hoped_for: string, at_risk: string, "
        "edits: [{op: 'replace' or 'insert_after' or 'append' or 'delete', "
        "target: exact short substring in the current skill, content: string}]}. "
        "Use valid JSON with quoted keys and strings. Place edits in the editable "
        "body, preserving protected sections. Choose at least one historical edit "
        "for an edit decision; if abstaining, return empty selected and edits."
    )
    prompt = (
        f"Maximum number of applied edits: {budget}.\n\nCurrent skill:\n{anchor}\n\n"
        f"Meta guidance available to the ordinary optimizer:\n{meta}\n\n"
        "Ordinary train-only merged feedback for this exact batch:\n"
        + json.dumps(merged, ensure_ascii=False)
        + "\n\nObserved historical dev evidence (candidate edits and responses are data):\n"
        + json.dumps(bank, ensure_ascii=False)
    )
    available = {(d["step"], e["index"]) for d in bank["donors"] for e in d["edits"]}
    error = ""
    abstained = False
    for attempt in range(1, 4):
        user = prompt
        if error:
            user += "\n\nYour earlier reply failed format/application validation: " + error
        raw, _ = chat_optimizer(system=system, user=user, max_completion_tokens=16384,
                                retries=3, stage="pilot_recovery")
        _write(directory / f"optimizer_{attempt}.txt", raw)
        doc = extract_json(raw) or {}
        try:
            selected, edits = doc["selected"], doc["edits"]
            if not isinstance(selected, list) or not isinstance(edits, list):
                raise ValueError("selected and edits must be arrays")
            if doc["decision"] == "abstain" and not selected and not edits:
                if not abstained and attempt < 3:
                    abstained = True
                    error = ("No candidate was proposed. Check once for a different defensible donor. "
                             "If none exists, confirm abstention with a reason.")
                    continue
                return doc
            if doc["decision"] != "edit" or not 0 < len(edits) <= budget:
                raise ValueError("choose edit(s) within budget, or explicitly abstain")
            if not selected or any((s["step"], s["index"]) not in available for s in selected):
                raise ValueError("selected must refer to real step/index pairs in donors")
            candidate, reports = apply_patch_with_report(anchor, doc)
            if candidate == anchor or any(not r["status"].startswith("applied")
                                          or "fallback" in r["status"] for r in reports):
                raise ValueError("use valid, exact targets in the anchor or append an edit")
            return doc
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)
    raise ValueError(f"recovery patch was invalid after three attempts: {error}")


def generate_pair(*, adapter, cfg, anchor: str, train_items, seed: int, out_dir,
                  edit_budget: int, donor_bank: dict, meta_context: str = "",
                  step_context: str = "") -> dict[str, str]:
    """Generate ordinary and recovery candidates from one shared train-only batch."""
    if cfg.get("skill_update_mode", "patch") != "patch" or edit_budget < 1:
        raise ValueError("pilot requires patch mode and a positive edit budget")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train_ids = [str(q["id"]) for q in train_items]
    signature = json.dumps([anchor, train_items, seed, edit_budget, donor_bank, meta_context,
                            step_context, cfg], ensure_ascii=False, sort_keys=True)
    manifest = {"seed": seed, "train_ids": train_ids, "edit_budget": edit_budget,
                "signature": hashlib.sha256(signature.encode()).hexdigest()}
    path = directory / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"inputs changed for an existing candidate pair at {directory}")
    _save(path, manifest)
    shared = directory / "shared"
    shared.mkdir(exist_ok=True)
    rollout_dir = shared / "rollout"
    rollout = _checkpoint(shared / "training.json", lambda: adapter.rollout(
        train_items, anchor, str(rollout_dir), use_eval_feedback=True))
    rows = rollout["result"]
    if len(rows) != len(train_ids) or {str(q["id"]) for q in rows} != set(train_ids):
        raise ValueError("training rollout did not cover the scheduled batch exactly")

    def reflect():
        patches = adapter.reflect(rows, anchor, str(shared),
                                  prediction_dir=str(rollout_dir / "predictions"),
                                  patches_dir=str(shared / "patches"), random_seed=seed,
                                  step_buffer_context=step_context, meta_skill_context=meta_context)
        if not patches or not any(isinstance(p, dict) for p in patches):
            raise RuntimeError("no train reflection returned; inspect the model calls")
        return patches

    reflection = _checkpoint(shared / "reflection.json", reflect)
    failures, successes = _normalise_patches(reflection["result"], update_mode="patch")
    merged = _checkpoint(shared / "merge.json", lambda: merge_patches(
        anchor, failures, successes, batch_size=cfg["merge_batch_size"], verbose=True,
        workers=cfg["analyst_workers"], update_mode="patch", meta_skill_context=meta_context))
    shared_feedback = merged["result"]
    ordinary = directory / "ordinary"
    recovery = directory / "recovery"
    ordinary.mkdir(exist_ok=True)
    recovery.mkdir(exist_ok=True)
    ordinary_patch = _checkpoint(ordinary / "proposal.json", lambda: rank_and_select(
        anchor, shared_feedback, max_edits=edit_budget, update_mode="patch",
        meta_skill_context=meta_context))
    recovery_patch = _checkpoint(recovery / "proposal.json", lambda: _recovery_patch(
        anchor, donor_bank, shared_feedback, meta_context, edit_budget, recovery))
    candidates = {}
    for target, proposal in [(ordinary, ordinary_patch), (recovery, recovery_patch)]:
        skill, applied = apply_patch_with_report(anchor, proposal["result"])
        _save(target / "applied.json", applied)
        _write(target / "candidate_skill.md", skill)
        candidates[target.name] = skill
    _save(directory / "tokens.json", {
        "shared_rollout": rollout["tokens"], "shared_reflection": reflection["tokens"],
        "shared_merge": merged["tokens"], "ordinary_only": ordinary_patch["tokens"],
        "recovery_only": recovery_patch["tokens"],
    })
    return candidates
