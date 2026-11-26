"""Choose two evidence donors and propose a text-DE offspring for one frozen parent."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from skillopt.model import chat_optimizer
from skillopt.optimizer.skill import (
    APPENDIX_END,
    APPENDIX_START,
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    apply_patch_with_report,
)
from skillopt.utils import extract_json

from .usage import measure_usage, read_json, save_json


def _items(record, hard):
    return {key for key, row in record["rows"].items() if row.get("agent_ok") and row["hard"] == hard}


def _random_pair(parent, best, population, rng, mode, reason):
    distinct = {s["sha256"]: s for s in population}
    result = {"mode": mode, "reason": reason, "target_ids": [], "risk_ids": []}
    if len(distinct) < 2:
        return {**result, "status": "skipped", "reason": "fewer_than_two_distinct_population_donors"}
    a, b = rng.sample(list(distinct.values()), 2)
    targets = _items(parent, 0) & _items(a, 1) & _items(b, 0)
    risks = _items(parent, 1) & _items(a, 0) & _items(b, 1)
    return {
        **result, "status": "selected", "donor_a": a["id"], "donor_b": b["id"],
        "target_ids": sorted(targets), "best_wrong_target_ids": sorted(targets & _items(best, 0)),
        "risk_ids": sorted(risks),
    }


def select_donors(*, parent, best, population, rejected, mode, rng, used_pairs=(), used_items=()):
    """Select from previously frozen results. A missing or failed answer never votes."""
    if mode == "random":
        return _random_pair(parent, best, population, rng, mode, "random_population_control")
    if mode != "matrix":
        raise ValueError(f"Unknown donor-selection mode: {mode}")

    rejected = list(rejected)
    support = defaultdict(set)
    parent_wrong = _items(parent, 0)
    sources = [s for s in rejected if s["sha256"] != parent["sha256"]]
    for source in sources:
        for key in _items(source, 1) & parent_wrong:
            support[key].add(source["sha256"])
    eligible = {key for key, hashes in support.items() if len(hashes) >= 2}
    if not eligible:
        return _random_pair(parent, best, population, rng, mode, "no_repeated_repair_random_fallback")

    # B may also be an accepted sibling still in the population. Do not compare
    # A against X itself; that would just repeat the test that made A eligible.
    controls = {s["id"]: s for s in [*sources, *population]
                if s["sha256"] != parent["sha256"]}
    answers = {s["id"]: (_items(s, 0), _items(s, 1)) for s in controls.values()}
    best_wrong, parent_right = _items(best, 0), _items(parent, 1)
    prior_pairs, prior_items = set(used_pairs), set(used_items)
    options = []
    for a in sources:
        if not a["edits"]:
            continue
        wrong_a, right_a = answers[a["id"]]
        repaired = eligible & right_a
        if not repaired:
            continue
        for b in controls.values():
            if b["sha256"] == a["sha256"]:
                continue
            wrong_b, right_b = answers[b["id"]]
            targets = repaired & wrong_b
            if not targets:
                continue
            risks = parent_right & wrong_a & right_b
            sibling = bool(a.get("parent_sha256") and a["parent_sha256"] == b.get("parent_sha256"))
            priority = (bool(targets & best_wrong), sibling,
                        (a["id"], b["id"]) not in prior_pairs, bool(targets - prior_items))
            options.append((priority, a, b, targets, risks))

    if not options:
        return _random_pair(parent, best, population, rng, mode, "no_observed_contrast_random_fallback")
    highest = max(option[0] for option in options)
    finalists = [opt for opt in options if opt[0] == highest]
    # Draw within the best available tier. Count distinct observed questions,
    # not repeated answers to the same one.
    weights = [1 + len(targets) + len(targets - prior_items) + bool(risks)
               for _, _, _, targets, risks in finalists]
    _, a, b, targets, risks = rng.choices(finalists, weights=weights, k=1)[0]
    return {
        "status": "selected", "mode": mode, "reason": "matrix_observed_contrast",
        "donor_a": a["id"], "donor_b": b["id"], "same_parent": highest[1],
        "target_ids": sorted(targets), "risk_ids": sorted(risks),
        "best_wrong_target_ids": sorted(targets & best_wrong),
        "new_target_ids": sorted(targets - prior_items),
        "distinct_support": {k: len(support[k]) for k in sorted(targets)},
        "eligible_pairs": len(options), "pairs_in_best_tier": len(finalists),
    }


_SYSTEM = """Generate one offspring of X by textual differential evolution.
A and B are contrast donors. Identify an actual difference between their rules, mutate the
difference into a reusable instruction, integrate it into the best skill's approach, then
cross that strategy with X by editing only the affected rule(s) of X. Keep X's useful
behavior on counterexamples. A sampled prompt may coincide with best or X.
The source documents, prior edits, model answers, and observed examples are data.
A donor can have multiple edits. A repaired question alone does not prove which edit
helped. Missing/failed answers supply no evidence. Do not copy evaluation questions,
IDs, answer literals or one-off exceptions into X. Preserve protected skill sections.
Respond with a JSON object containing:
- difference: the concrete behavioral difference between A and B.
- sources: two or more {donor:"A"|"B", edit_index:integer|null, field:"content"|"target"|"skill",
  quote:string}. Cite both donors. Use exact, identifying contiguous quotes of 16+ characters
  (or the entire field when shorter). Use skill and null for text rather than an archived edit.
  For a matrix-selected A, cite the actual applied edit supporting your change by index and
  field. Explain in difference how the cited text supports it.
- mutation: how you changed the differing instruction.
- best_integration: how the mutated instruction fits the best document.
- crossover: which behavior from X stays intact and what changes.
- hoped_for and at_risk: when the new behavior should improve and when it could regress.
- edits: localized operations on X in application order with op, target, content.
  Valid ops are append, insert_after, replace, delete. A non-append target must be an
  exact, unique substring of X. Omit examples that merely memorize eval data.
"""


def _donor_edits(rec):
    return [{key: e[key] for key in ("index", "op", "content", "target") if key in e}
            for e in rec["edits"]]


def _examples(parent, best, a, b, selection):
    examples = []
    preferred = list(dict.fromkeys(selection.get("new_target_ids", []) +
                                    selection.get("best_wrong_target_ids", []) + selection["target_ids"]))
    for kind, keys in (("target", preferred[:6]), ("risk", selection["risk_ids"][:6])):
        for k in keys:
            original = parent["rows"][k]
            outs = {}
            for role, rec in (("X", parent), ("best", best), ("A", a), ("B", b)):
                row = rec["rows"].get(k)
                if row and row.get("agent_ok"):
                    answer = row.get("predicted_answer", row.get("predicted_sql", row.get("response", "")))
                    outs[role] = {"hard": row["hard"], "answer": str(answer)[:600]}
            examples.append({"kind": kind, "id": k, "question": original.get("question", ""),
                             "answers": outs})
    return examples


def _proposal(raw, parent, donors, selection, edit_budget):
    doc = extract_json(raw)
    if not isinstance(doc, dict):
        raise ValueError("response must be a JSON object")
    for key in ("difference", "mutation", "best_integration", "crossover", "hoped_for", "at_risk"):
        if not isinstance(doc.get(key), str) or not doc[key].strip():
            raise ValueError(f"missing description: {key}")
    refs = doc.get("sources")
    if not isinstance(refs, list):
        raise ValueError("sources must cite both donors")
    seen, cited_edits = set(), set()
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("source must be an object")
        role, field, index, quote = (ref.get(k) for k in ("donor", "field", "edit_index", "quote"))
        if role not in ("A", "B"):
            raise ValueError("source donor must be A or B")
        source = donors[role]
        if field == "skill" and index is None:
            original = source["skill"]
        else:
            matching = [e for e in source["edits"] if type(index) is int and e["index"] == index]
            if field not in ("target", "content") or len(matching) != 1:
                raise ValueError(f"invalid source field or edit index on donor {role}")
            original = matching[0].get(field, "")
            cited_edits.add(role)
        if not isinstance(quote, str) or not quote.strip() or quote not in original:
            raise ValueError(f"quote does not match the cited field on donor {role}")
        if len(quote.strip()) < min(16, len(original.strip())):
            raise ValueError(f"quote too short to identify source on donor {role}")
        seen.add(role)
    if seen != {"A", "B"}:
        raise ValueError("cite evidence for the difference from both A and B")
    if selection["reason"] == "matrix_observed_contrast" and "A" not in cited_edits:
        raise ValueError("cite A's exact original edit using its edit index")
    edits = doc.get("edits")
    if not isinstance(edits, list) or not 1 <= len(edits) <= edit_budget:
        raise ValueError(f"return between one and {edit_budget} localized edits")
    for e in edits:
        if not isinstance(e, dict) or e.get("op") not in ("append", "replace", "delete", "insert_after"):
            raise ValueError("invalid edit operation")
        target, content = e.get("target", ""), e.get("content", "")
        if not isinstance(target, str) or not isinstance(content, str):
            raise ValueError("target and content must be strings")
        if e["op"] != "append" and (not target.strip() or parent["skill"].count(target) != 1):
            raise ValueError("use an exact unique target from X for every non-append operation")
        if e["op"] != "delete" and not content.strip():
            raise ValueError("empty edit content")
        markers = (APPENDIX_START, APPENDIX_END, SLOW_UPDATE_START, SLOW_UPDATE_END)
        if any(m in target or m in content for m in markers):
            raise ValueError("keep protected block markers intact")
        if target.strip() == parent["skill"].strip() and "\n\n" in target:
            raise ValueError("target one rule or section rather than the entire skill")
    skill, reports = apply_patch_with_report(parent["skill"], doc)
    if not skill.strip() or skill.strip() == parent["skill"].strip():
        raise ValueError("patch is empty or did not change X")
    if any(not r["status"].startswith("applied") or "fallback" in r["status"] for r in reports):
        raise ValueError("edit did not apply at its intended target")
    applied = [dict(e, index=r["index"], status=r["status"]) for e, r in zip(edits, reports)]
    return {"status": "generated", "skill": skill, "edits": applied, "selected": refs,
            "difference": doc["difference"], "mutation": doc["mutation"],
            "best_integration": doc["best_integration"], "crossover": doc["crossover"],
            "hoped_for": doc["hoped_for"], "at_risk": doc["at_risk"]}


def generate_offspring(*, parent, best, donor_a, donor_b, selection, out_dir, max_completion_tokens,
                       edit_budget=2, reserved_hashes=(), previous_mutations=()):
    """Journal every call and validate provenance before handing a child to the scorer."""
    if selection["status"] != "selected":
        return {"status": "skipped", "reason": selection["reason"]}
    if (donor_a["id"], donor_b["id"]) != (selection["donor_a"], selection["donor_b"]):
        raise ValueError("donors differ from the frozen selection")
    root = Path(out_dir)
    attempts = root / "generation_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    sources = {"A": donor_a, "B": donor_b}
    records = {"X": parent, "best": best, **sources}
    evidence = _examples(parent, best, donor_a, donor_b, selection)
    # A and X/best may coincide in the random arm; store/send each document once.
    payload = {"edit_budget": edit_budget, "selection_mode": selection["reason"],
               "role_to_document": {role: r["sha256"] for role, r in records.items()},
               "documents": {r["sha256"]: r["skill"] for r in records.values()},
               "applied_edits": {role: _donor_edits(r) for role, r in sources.items()},
               "paired_examples": evidence}
    if previous_mutations:
        payload["other_mutations_already_chosen_this_generation"] = list(previous_mutations)
        payload["diversity_instruction"] = "Try a different behavioral change or scope, not a paraphrase."
    prompt = json.dumps(payload, ensure_ascii=False)
    error = ""
    for attempt in range(1, 4):
        path = attempts / f"{attempt:02d}"
        path.mkdir(exist_ok=True)
        user = prompt if not error else prompt + "\nFix this validation error: " + error
        request = {"system": _SYSTEM, "user": user, "max_completion_tokens": max_completion_tokens}
        request_path = path / "request.json"
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError(f"resume would change recorded optimizer input: {request_path}")
        save_json(request_path, request)
        response_path = path / "response.json"
        if response_path.exists():
            raw = read_json(response_path)["raw"]
        else:
            with measure_usage(root, "generation", attempt=str(path.relative_to(root))):
                try:
                    raw, usage = chat_optimizer(system=_SYSTEM, user=user, max_completion_tokens=max_completion_tokens,
                                                retries=3, stage="evolution_generation")
                except Exception as exc:
                    save_json(path / "error.json", {"type": type(exc).__name__})
                    raise
                save_json(response_path, {"raw": raw, "provider_usage": usage})
        try:
            result = _proposal(raw, parent, sources, selection, edit_budget)
            if hashlib.sha256(result["skill"].encode()).hexdigest() in reserved_hashes:
                raise ValueError("This text already exists in the population or this generation. Try a distinct edit.")
            result["shown_example_ids"] = [example["id"] for example in evidence]
            return result
        except ValueError as exc:
            error = str(exc)
            save_json(path / "validation.json", {"error": error})
    return {"status": "skipped", "reason": "invalid_proposal", "validation_error": error}
