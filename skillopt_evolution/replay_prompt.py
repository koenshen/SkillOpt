"""Replay an earlier evolution run's fixed DE assignments with the current generator."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .archive import load_archive
from .evaluate import evaluate_candidate
from .evolve import _donor_edits, _examples, generate_offspring
from .run import ROOT, archived_adapter, check_or_write, fingerprint, safe_name, sha
from .usage import now, read_json, save_json, summarize_usage, zero_usage


def compare_examples(examples, old_rows, new_rows, parent_rows, skipped):
    """One slot and one exposure per displayed example, on the old observed set."""
    comparison = {}
    for kind in ("target", "risk"):
        stats = dict(n=0, old_hits=0, new_hits=0, new_measured=0, fallback_to_parent=0,
                     excluded_old_unobserved=[], rows=[])
        for example in examples:
            if example["kind"] != kind:
                continue
            key = example["id"]
            old = old_rows.get(key)
            if not old or not old["agent_ok"]:
                stats["excluded_old_unobserved"].append(key)
                continue
            new = parent_rows[key] if skipped else new_rows[key]
            if not new["agent_ok"]:
                raise ValueError(f"Incomplete paired result for {key}")
            stats["n"] += 1
            stats["old_hits"] += int(old["hard"])
            stats["new_hits"] += int(new["hard"])
            stats["new_measured"] += int(not skipped)
            stats["fallback_to_parent"] += int(skipped)
            stats["rows"].append(dict(id=key, question=example["question"],
                                      old_answer=old.get("predicted_answer"), old_hit=old["hard"],
                                      new_answer=new.get("predicted_answer"), new_hit=new["hard"],
                                      origin="archived_parent" if skipped else "child"))
        comparison[kind] = stats
    return comparison


def _original_input(old_slot, records, plan):
    old_request = read_json(old_slot / "generation_attempts/01/request.json")
    original = json.loads(old_request["user"])
    roles = {role: records[plan[key]] for role, key in
             (("X", "parent"), ("best", "best"), ("A", "donor_a"), ("B", "donor_b"))}
    expected = {role: rec["sha256"] for role, rec in roles.items()}
    if original["role_to_document"] != expected or any(
        original["documents"].get(rec["sha256"]) != rec["skill"] for rec in roles.values()
    ):
        raise ValueError(f"Archived donor/parent text mismatch: {old_slot}")
    if original["applied_edits"] != {role: _donor_edits(roles[role]) for role in ("A", "B")}:
        raise ValueError(f"Archived donor edits mismatch: {old_slot}")
    examples = _examples(roles["X"], roles["best"], roles["A"], roles["B"], plan)
    if examples != original["paired_examples"]:
        raise ValueError(f"Displayed evidence differs from the old prompt: {old_slot}")
    return original, roles


def replay_slot(*, adapter, items, records, generation_dir, old_slot, work, plan, frozen,
                edit_budget, reserved_hashes):
    old_decision = read_json(old_slot / "decision.json")
    original, roles = _original_input(old_slot, records, plan)
    if original["edit_budget"] != edit_budget:
        raise ValueError(f"Original edit budget differs: {old_slot}")
    old_eval = read_json(old_slot / "evaluation_inputs.json")["signature"]
    if (old_eval["frozen_order"] != frozen["order"] or old_decision["parent"] != plan["parent"]
            or old_decision["parent_correct"] != roles["X"]["correct"]
            or old_decision["sha256"] != old_eval["child_sha256"]):
        raise ValueError(f"Mismatch in frozen gate or archived child: {old_slot}")
    check_or_write(work / "replay_inputs.json", {
        "original_slot": str(old_slot), "source_frozen": str(generation_dir / "frozen.json"),
        "plan": plan, "original_payload_sha256": fingerprint(original),
        "old_decision_sha256": fingerprint(old_decision),
        "previous_mutations": original.get("other_mutations_already_chosen_this_generation", []),
    })
    pp = work / "proposal.json"
    if pp.is_file():
        proposal = read_json(pp)
    else:
        proposal = generate_offspring(parent=roles["X"], best=roles["best"], donor_a=roles["A"],
                                      donor_b=roles["B"], selection=plan, out_dir=work / "generation",
                                      max_completion_tokens=adapter.max_completion_tokens, edit_budget=edit_budget,
                                      reserved_hashes=reserved_hashes,
                                      previous_mutations=original.get("other_mutations_already_chosen_this_generation", []))
        save_json(pp, proposal)
    skipped = proposal["status"] == "skipped"
    parent = roles["X"]
    examples = original["paired_examples"]
    previous = old_decision["rows"]
    rows = {}
    gate_summary = None
    if not skipped:
        skill = proposal["skill"]
        (work / "child_skill.md").write_text(skill, encoding="utf-8")
        cache = {}
        sources = {}
        # A previously scored native text can have a complete or partial sheet.
        for rec in sorted(records.values(), key=lambda rec: not rec["complete"]):
            if rec["sha256"] == sha(skill):
                for k, v in rec["rows"].items():
                    cache.setdefault(k, v)
                    sources.setdefault(k, rec["id"])
        gate = evaluate_candidate(adapter=adapter, dev_items=items, order=frozen["order"], skill=skill,
                                  parent_rows=parent["rows"], cached_rows=cache, cache_sources=sources,
                                  out_dir=work / "gate", batch_size=old_eval["batch_size"])
        gate_summary = {k:gate[k] for k in ("decision", "accepted", "n_observed", "correct",
                                            "accuracy", "upper_bound", "fixes", "regressions")}
        rows = dict(gate["rows"])
        # Fix the comparison denominator first. Probe only old evaluated,
        # displayed IDs the new gate hasn't reached. Never revise that gate.
        probe_ids = list(dict.fromkeys(ex["id"] for ex in examples if
                         previous.get(ex["id"], {}).get("agent_ok") and ex["id"] not in rows))
        if probe_ids:
            by_id = {str(x["id"]): x for x in items}
            probe = evaluate_candidate(adapter=adapter, dev_items=[by_id[k] for k in probe_ids],
                                       order=probe_ids, skill=skill,
                                       parent_rows={k:parent["rows"][k] for k in probe_ids},
                                       cached_rows={}, out_dir=work / "probe",
                                       batch_size=old_eval["batch_size"], require_complete=True)
            rows.update(probe["rows"])
    comparison = compare_examples(examples, previous, rows, parent["rows"], skipped)
    result = {"parent": parent["id"], "best": plan["best"], "donors": [plan["donor_a"], plan["donor_b"]],
              "archived_parent_correct": parent["correct"], "archived_best_correct": roles["best"]["correct"],
              "proposal_status": proposal["status"], "skip_reason": proposal.get("reason"),
              "beats_global_best": bool(gate_summary and gate_summary["accuracy"] is not None
                                        and gate_summary["correct"] > roles["best"]["correct"]),
              "gate": gate_summary, "old_gate": old_decision["decision"], "comparison": comparison,
              "usage": {stage:summarize_usage(work / stage)["total"] for stage in ("generation", "gate", "probe")}}
    save_json(work / "comparison.json", result)
    return result


def summarize(slots, expected):
    stats = {}
    for kind in ("target", "risk"):
        keys = ("n", "old_hits", "new_hits", "new_measured", "fallback_to_parent")
        stats[kind] = {key:sum(s["comparison"][kind][key] for s in slots) for key in keys}
        stats[kind]["measured_child_hits"] = sum(r["new_hit"] for s in slots
                                                for r in s["comparison"][kind]["rows"] if r["origin"] == "child")
    usage = {}
    for stage in ("generation", "gate", "probe"):
        usage[stage] = {k:sum(s["usage"][stage][k] for s in slots) for k in zero_usage()}
    return dict(status="complete" if len(slots) == expected else "running", expected_slots=expected,
                finished_slots=len(slots), generated=sum(s["proposal_status"] == "generated" for s in slots),
                skipped=sum(s["proposal_status"] == "skipped" for s in slots),
                gate_wins=sum(bool(s["gate"] and s["gate"]["accepted"]) for s in slots),
                global_best_wins=sum(s["beats_global_best"] for s in slots),
                comparison=stats, usage=usage)


def replay_run(out, adapter, archive, items, meta, generations):
    records = {rec["id"]: rec for rec in archive["records"]}
    result = []
    expected = sum(len(frozen["plans"]) for _, frozen in generations)
    for generation_dir, frozen in generations:
        reserved = {x["sha256"] for x in frozen["parents"]}
        group = out / "generations" / generation_dir.name
        check_or_write(group / "frozen.json", frozen)
        for plan in frozen["plans"]:
            sl = f"slot_{plan['slot']:02d}"
            work = group / sl
            current = replay_slot(adapter=adapter, items=items, records=records,
                                  generation_dir=generation_dir, old_slot=generation_dir/sl, work=work,
                                  plan=plan, frozen=frozen, edit_budget=meta["options"]["edit_budget"],
                                  reserved_hashes=reserved)
            proposal = read_json(work / "proposal.json")
            if proposal["status"] == "generated":
                reserved.add(sha(proposal["skill"]))
            result.append(current)
            status = summarize(result, expected)
            status["reported_usage"] = summarize_usage(out)
            save_json(out / "summary.json", status)
            print(f"{generation_dir.name}/{sl} {proposal['status']}, targets "
                  f"{current['comparison']['target']['new_hits']}/{current['comparison']['target']['n']}, "
                  f"{status['finished_slots']}/{expected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prior-run", type=Path)
    group.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        out = args.resume.resolve()
        prior = Path(read_json(out / "manifest.json")["prior_run"])
    else:
        prior = args.prior_run.resolve()
    meta = read_json(prior / "manifest.json")
    if read_json(prior / "summary.json")["status"] != "complete":
        raise ValueError("Replay requires the earlier evolution to be complete")
    source = Path(meta["source"])
    os.chdir(source.parent.parent)
    cfg = read_json(source / "config.json")
    items = read_json(Path(meta["split_path"]))
    if fingerprint(cfg) != meta["config_sha256"] or fingerprint(items) != meta["split_sha256"]:
        raise ValueError("Source config or dev split changed")
    archive = load_archive(source, [str(row["id"]) for row in items])
    if fingerprint(archive["records"]) != meta["archive_sha256"]:
        raise ValueError("Source genealogy, edits or scores changed")
    records = {rec["id"]: rec for rec in archive["records"]}
    generations = [(p.parent, read_json(p)) for p in sorted(prior.glob("generations/*/frozen.json"))]
    if not generations:
        raise ValueError(f"No frozen plans in {prior}")
    old_hashes = []
    for g, frozen in generations:
        for p in frozen["plans"]:
            # A fixed contrast replays slots with a measured old child and
            # known historical parents. Never substitute a new parent or pair.
            _, roles = _original_input(g / f"slot_{p['slot']:02d}", records, p)
            if roles["best"]["sha256"] != frozen["best"]["sha256"]:
                raise ValueError(f"Unrecognized historical best in {g}")
            slot = g / f"slot_{p['slot']:02d}"
            old_hashes.append(fingerprint([read_json(slot/"decision.json"), read_json(
                slot/"generation_attempts/01/request.json"), p, frozen["order"]]))
    cfg.update(meta["source_config"])
    if not args.resume:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        name = (f"skillopt_{safe_name(meta['benchmark'])}_{safe_name(cfg['target_model'])}_{stamp}"
                f"_evo2_prompt_replay_rho{round(100*cfg['train_early_stop_rho']):03d}")
        out = prior.parent / name
        out.mkdir(mode=0o700)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature = {"prior_run": str(prior), "source": str(source), "source_archive": meta["archive_sha256"],
                     "split_sha256": fingerprint(items), "original_assignments_sha256": fingerprint(old_hashes),
                     "prior_manifest_sha256": fingerprint(meta), "source_config": meta["source_config"],
                     "implementation": {p.name: sha(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        manifest = out / "manifest.json"
        if manifest.exists():
            if any(read_json(manifest).get(k) != v for k,v in signature.items()):
                raise ValueError("Resume fingerprint mismatch")
        else:
            revision = subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
            save_json(manifest, dict(signature, created_utc=now(), git_revision=revision))
        print(f"Output: {out}", flush=True)
        try:
            replay_run(out, archived_adapter(cfg), archive, items, meta, generations)
        except Exception as exc:
            save_json(out / "last_failure.json", {"time":now(),"type":type(exc).__name__,
                                                   "reported_usage":summarize_usage(out)})
            raise


if __name__ == "__main__":
    main()
