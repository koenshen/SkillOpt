"""Run second-stage evolution from one finished QA archive, independently of training."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import inspect
import json
import os
import platform
import random
import re
import subprocess
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from skillopt.evaluation.early_stop import lineage_order, update_history

from .archive import load_archive
from .evaluate import evaluate_candidate
from .evolve import generate_offspring, select_donors
from .usage import now, read_json, save_json, summarize_usage

ROOT = Path(__file__).resolve().parents[1]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def winner(population):
    return max(population, key=lambda rec: rec["correct"])


def identity(record):
    return {k: record[k] for k in ("id", "sha256", "correct", "score")}


def matrix_file(path, records, ids):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["candidate", "sha256", "action", *ids])
        for rec in records:
            row = rec["rows"]
            w.writerow([rec["id"], rec["sha256"], rec["action"],
                        *(("ERR" if not row[k].get("agent_ok") else int(row[k]["hard"]))
                          if k in row else "None" for k in ids)])
    tmp.replace(path)


def frozen_selections(population, best, rejected, mode, seed, generation):
    plans, used_pairs, used_items = [], set(), set()
    for slot, parent in enumerate(population, 1):
        slot_seed = f"{seed}/{generation}/{slot}"
        plan = select_donors(parent=parent, best=best, population=population, rejected=rejected,
                             mode=mode, rng=random.Random(slot_seed), used_pairs=used_pairs,
                             used_items=used_items)
        plans.append(dict(plan, slot=slot, rng_seed=slot_seed, parent=parent["id"], best=best["id"]))
        if plan["status"] == "selected":
            used_pairs.add((plan["donor_a"], plan["donor_b"]))
            used_items.update(plan["target_ids"])
    return plans


def check_or_write(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Resume conflicts with frozen inputs at {path}")
    else:
        save_json(path, value)


def run_loop(*, archive, items, adapter, out, options):
    """Only this loop mutates stage-2 state. Replay committed slots on resume."""
    population = list(archive["population"])
    best = winner(population)
    records = list(archive["records"])
    by_id = {r["id"]: r for r in records}
    history = {k: list(values) for k, values in archive["history"].items()}
    dev_ids = [str(item["id"]) for item in items]
    cache, origins = defaultdict(dict), defaultdict(dict)
    # A selected full sheet owns its archived score, even if this text was
    # measured elsewhere. Do not manufacture a best-of-several score per item.
    for rec in [*population, *sorted(records, key=lambda r: not r["complete"])]:
        for key, row in rec["rows"].items():
            cache[rec["sha256"]].setdefault(key, row)
            origins[rec["sha256"]].setdefault(key, rec["id"])

    check_or_write(out / "initial_population.json", archive["initial_population"])
    save_json(out / "archive_audit.json", {"skipped": archive["skipped"],
              "native_step_usage": archive["native_step_usage"]})
    matrix_file(out / "matrix.tsv", records, dev_ids)
    flat = 0
    generations = []
    for generation in range(1, options["generations"] + 1):
        directory = out / "generations" / f"{generation:03d}"
        order = lineage_order(dev_ids, history)
        parents = list(population)
        rejected = [rec for rec in records if rec["action"] == "reject"]
        plans = frozen_selections(parents, best, rejected, options["strategy"], options["seed"], generation)
        check_or_write(directory / "frozen.json", {"parents": list(map(identity, parents)),
                       "best": identity(best), "order": order, "plans": plans})
        print(f"Generation {generation}: {len(parents)} parents, best {best['id']} "
              f"({best['correct']}/{len(dev_ids)})", flush=True)
        proposals = []
        reserved = {r["sha256"] for r in parents}

        # Finish generating every child before the first validation call.
        for slot, (parent, plan) in enumerate(zip(parents, plans), 1):
            work = directory / f"slot_{slot:02d}"
            path = work / "proposal.json"
            if path.exists():
                proposal = read_json(path)
            elif plan["status"] == "selected":
                proposal = generate_offspring(
                    parent=parent, best=best, donor_a=by_id[plan["donor_a"]],
                    donor_b=by_id[plan["donor_b"]], selection=plan, out_dir=work,
                    edit_budget=options["edit_budget"], max_completion_tokens=adapter.max_completion_tokens,
                    reserved_hashes=reserved,
                    previous_mutations=[p["mutation"] for p in proposals if p["status"] == "generated"])
            else:
                proposal = {"status": "skipped", "reason": plan["reason"]}
            if proposal["status"] == "generated":
                h = sha(proposal["skill"])
                if h in reserved:
                    proposal = dict(proposal, status="skipped", reason="duplicate_in_frozen_generation")
                reserved.add(h)
            save_json(path, proposal)
            if proposal["status"] == "generated":
                (work / "child_skill.md").write_text(proposal["skill"], encoding="utf-8")
            proposals.append(proposal)

        replacements = 0
        results = []
        next_population = list(parents)
        for slot, (parent, proposal) in enumerate(zip(parents, proposals), 1):
            work = directory / f"slot_{slot:02d}"
            receipt = work / "decision.json"
            uid = f"g{generation:03d}s{slot:02d}"
            if proposal["status"] == "skipped":
                results.append({"id": uid, "parent": parent["id"], "action": "skipped", "reason": proposal["reason"],
                                "usage": summarize_usage(work)})
                check_or_write(receipt, results[-1])
                continue
            text = proposal["skill"]
            h = sha(text)
            if receipt.exists():
                decision = read_json(receipt)
            else:
                decision = evaluate_candidate(
                    adapter=adapter, dev_items=items, order=order, skill=text,
                    parent_rows=parent["rows"], cached_rows=cache[h], cache_sources=origins[h],
                    out_dir=work, batch_size=4)
                decision.update(id=uid, parent=parent["id"], parent_correct=parent["correct"], sha256=h)
                shown = set(proposal.get("shown_example_ids", []))
                decision["fixes_outside_shown_examples"] = [k for k in decision["fixes"] if k not in shown]
                decision["regressions_outside_shown_examples"] = [k for k in decision["regressions"] if k not in shown]
                save_json(receipt, decision)
            if decision["parent"] != parent["id"] or decision["sha256"] != h:
                raise ValueError(f"Cached decision no longer matches {uid}")
            rows = decision["rows"]
            accepted = decision["accepted"]
            if accepted and (not decision["complete"] or decision["correct"] <= parent["correct"]):
                raise ValueError(f"Invalid replacement: {uid}")
            action = "accepted" if accepted else "reject"
            child = dict(id=uid, skill=text, sha256=h, parent_sha256=parent["sha256"],
                         action=action, complete=decision["complete"], rows=rows, step=None,
                         correct=decision["correct"] if decision["complete"] else None,
                         score=decision["accuracy"], edits=proposal["edits"], source=str(work))
            by_id[uid] = child
            records.append(child)
            for key in decision["new_ids"]:
                cache[h][key] = rows[key]
                origins[h][key] = uid
            update_history(history, [rows[key] for key in decision["new_ids"]])
            if accepted:
                next_population[slot-1] = child
                replacements += 1
            results.append(dict(id=uid, parent=parent["id"], action=action, n=decision["n_observed"],
                                score=decision["accuracy"], upper_bound=decision["upper_bound"],
                                fixes=len(decision["fixes"]), regressions=len(decision["regressions"]),
                                usage=summarize_usage(work)))
            print(f"  {uid} {action} vs {parent['id']}, observed {decision['n_observed']}, "
                  f"upper={decision['upper_bound']:.4f}", flush=True)

        population = next_population
        # Preserve the incumbent on a tie for the global best.
        top = winner(population)
        if top["correct"] > best["correct"]:
            best = top
        flat = 0 if replacements else flat + 1
        generation_summary = dict(generation=generation, replacements=replacements, flat_generations=flat,
                                  best=identity(best), population=list(map(identity, population)), slots=results)
        save_json(directory / "summary.json", dict(generation_summary, usage=summarize_usage(directory)))
        generations.append(generation_summary)
        matrix_file(out / "matrix.tsv", records, dev_ids)
        tmp_best = out / "best_skill.tmp"
        tmp_best.write_text(best["skill"], encoding="utf-8")
        tmp_best.replace(out / "best_skill.md")
        reason = None
        if flat >= options["patience"]:
            reason = "consecutive_generations_without_replacement"
        elif generation == options["generations"]:
            reason = "generation_limit"
        incremental = summarize_usage(out)
        combined = {k: archive["native_step_usage"][k] + incremental["total"][k]
                    for k in ("input_tokens", "output_tokens", "total_tokens")}
        summary = dict(status="complete" if reason else "running", stop_reason=reason,
                       initial_best=identity(winner(archive["population"])), best=identity(best),
                       gain_correct_over_initial_best=best["correct"] - winner(archive["population"])["correct"],
                       native_step_usage=archive["native_step_usage"], incremental_usage=incremental,
                       native_steps_plus_evolution_usage=combined, generations=generations)
        save_json(out / "summary.json", summary)
        if reason:
            return summary


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def archived_adapter(cfg):
    from skillopt.envs.base import EnvAdapter
    from skillopt.envs.bird.adapter import BirdAdapter
    from skillopt.envs.searchqa.adapter import SearchQAAdapter
    from skillopt.model import configure_openai_compatible, set_backend, set_reasoning_effort

    if cfg.get("env") not in {"bird", "searchqa"}:
        raise ValueError(f"Unsupported archived adapter: {cfg.get('env')}")
    for role in ["optimizer", "target"]:
        if cfg.get(role + "_backend") != "openai_compatible":
            raise ValueError(f"This runner requires the archived OpenAI-compatible {role} backend")
    set_backend("openai_compatible")
    configure_openai_compatible(optimizer_model=cfg["optimizer_model"], target_model=cfg["target_model"],
                                max_tokens=int(cfg["max_completion_tokens"]))
    set_reasoning_effort(cfg.get("reasoning_effort"))
    cls = BirdAdapter if cfg["env"] == "bird" else SearchQAAdapter
    adapter = cls(**{k: cfg[k] for k in inspect.signature(cls).parameters if k in cfg})
    # Stock split setup loads all three partitions. The adapter needs only the
    # explicit dev items and its environment settings in this stage.
    EnvAdapter.setup(adapter, cfg)
    return adapter


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path, help="one completed first-stage run")
    group.add_argument("--resume", type=Path, help="existing second-stage output; reload its options")
    p.add_argument("--strategy", choices=["matrix", "random"], default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--inspect", action="store_true", help="read-only archive and first-generation plan, no API calls")
    args = p.parse_args(argv)
    if args.resume:
        if any(v is not None for v in (args.strategy, args.generations, args.patience, args.seed)):
            p.error("Resume uses the original run's options")
        out = args.resume.resolve()
        prior = read_json(out / "manifest.json")
        source, options = Path(prior["source"]), prior["options"]
    else:
        source = args.source.resolve()
    # Native configs store repo-relative data and database paths. Output roots
    # sit directly beside the first-stage run, even when invoked from elsewhere.
    source_root = source.parent.parent
    os.chdir(source_root if (source_root / "data").is_dir() else ROOT)
    cfg = read_json(source / "config.json")
    split = Path(cfg["split_dir"])
    bench = split.name.removesuffix("_split")
    items_path = split / "val" / "items.json"
    items = read_json(items_path)
    ids = [str(q["id"]) for q in items]
    if not cfg.get("selection_early_stop") or not cfg.get("train_early_stop"):
        raise ValueError("Expected a source with dev and training early stopping enabled")
    archive = load_archive(source, ids)
    if not args.resume:
        options = {"strategy": args.strategy or "matrix", "seed": cfg["seed"] if args.seed is None else args.seed,
                   "generations": 3 if args.generations is None else args.generations,
                   "patience": 2 if args.patience is None else args.patience,
                   "edit_budget": int(read_json(source / "history.json")[-1].get("edit_budget") or cfg["edit_budget"])}
    if min(options["generations"], options["patience"], options["edit_budget"]) < 1:
        p.error("Generations, patience and edit budget must be positive")
    if args.inspect:
        plans = frozen_selections(archive["population"], winner(archive["population"]),
                                  [r for r in archive["records"] if r["action"] == "reject"],
                                  options["strategy"], options["seed"], 1)
        print(json.dumps(dict(source=str(source), selected=list(map(identity, archive["population"])),
                              shortfall=archive["initial_population"]["shortfall"],
                              excluded_by_cap=archive["initial_population"]["excluded_by_cap"],
                              native_step_usage=archive["native_step_usage"], first_generation=plans),
                         ensure_ascii=False, indent=2))
        return

    rho = float(cfg["train_early_stop_rho"])
    if not args.resume:
        started = datetime.now(ZoneInfo("Asia/Shanghai"))
        for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M%S_%f"):
            stamp = started.strftime(fmt)
            name = f"skillopt_{safe_name(bench)}_{safe_name(cfg['target_model'])}_{stamp}_evo2_rho{round(100*rho):03d}"
            out = source.parent / name
            try:
                out.mkdir(mode=0o700)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"Could not reserve a new output folder next to {source}")
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model_fields = ("env", "optimizer_model", "target_model", "optimizer_backend", "target_backend",
                        "reasoning_effort", "workers", "max_completion_tokens", "train_early_stop_rho")
        # Include scores, genealogy, applied text and actual answers used for donor selection.
        manifest = {"format": 1, "source": str(source), "output": str(out), "benchmark": bench, "options": options,
                    "split_path": str(items_path.resolve()), "split_sha256": fingerprint(items),
                    "archive_sha256": fingerprint(archive["records"]), "config_sha256": fingerprint(cfg),
                    "source_config": {k: cfg.get(k) for k in model_fields},
                    "population_cap": 5, "screen_batch": 4, "dev_size": len(ids),
                    "implementation_sha256": {s.name: sha(s.read_text()) for s in sorted(Path(__file__).parent.glob("*.py"))}}
        if args.resume:
            if any(prior.get(k) != v for k, v in manifest.items()):
                raise ValueError("Recorded source, implementation or settings changed; cannot mix results")
            previous_summary = out / "summary.json"
            if previous_summary.exists():
                finished = read_json(previous_summary)
                if finished.get("status") == "complete":
                    if sha((out / "best_skill.md").read_text()) != finished["best"]["sha256"]:
                        raise ValueError(f"Final skill does not match {previous_summary}")
                    print(f"Already complete: {previous_summary}")
                    return
        else:
            revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      check=False, capture_output=True, text=True)
            save_json(out / "manifest.json", dict(manifest, created_utc=now(), python=platform.python_version(),
                                                   git_revision=revision.stdout.strip()))
        print(f"Output: {out}", flush=True)
        try:
            result = run_loop(archive=archive, items=items, adapter=archived_adapter(cfg), out=out, options=options)
        except Exception as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            save_json(out / "last_failure.json", {"time": now(), "exception": type(exc).__name__,
                                                  "stack": [f"{f.filename}:{f.lineno} {f.name}" for f in frames],
                                                  "incremental_usage": summarize_usage(out)})
            raise
        print(f"Finished: {result['stop_reason']}; best={result['best']}; "
              f"stage-2 tokens={result['incremental_usage']['total']}", flush=True)


if __name__ == "__main__":
    main()
