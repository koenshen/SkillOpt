"""Wait for one completed evolution run and score its frozen best on TriviaQA test."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from skillopt_evolution.run import archived_adapter, fingerprint
from skillopt_evolution.usage import now, read_json, save_json

from .scoring import run_test


def wait_for_winner(run: Path, interval: float = 30):
    # Completion is written atomically *after* best_skill.md. While native
    # evolution holds its lock, even a previous failure marker may be stale.
    summary = run / "summary.json"
    while True:
        if summary.exists():
            state = read_json(summary)
            if state.get("status") == "complete":
                raw = (run / "best_skill.md").read_bytes()
                if hashlib.sha256(raw).hexdigest() != state["best"]["sha256"]:
                    raise ValueError("Final skill differs from evolution summary hash")
                return state, raw.decode("utf-8")
        with (run / ".run.lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                # The evolution could have published its final result between
                # reading summary.json and our acquiring the lock.
                if summary.exists() and read_json(summary).get("status") == "complete":
                    continue
                failure = run / "last_failure.json"
                cause = read_json(failure).get("exception") if failure.exists() else "process exited"
                raise RuntimeError(f"Evolution stopped without a final result ({cause}): {run}")
        time.sleep(interval)


def first_stage_reference(source, items):
    sheet = source / "test_eval" / "results.jsonl"
    if not sheet.exists():
        return {"status": "unavailable", "path": str(sheet)}
    with sheet.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by_id = {str(x["id"]): x for x in rows}
    if len(rows) != len(items) or set(by_id) != {str(x["id"]) for x in items}:
        raise ValueError(f"First-stage reference does not match the fixed test IDs: {sheet}")
    for item in items:
        row = by_id[str(item["id"])]
        answers = row.get("gold_answers", row.get("gold_answer", []))
        if row["question"] != item["question"] or set(answers) != set(item["answers"]):
            raise ValueError(f"Reference question or answers changed for {item['id']}")
    return {"status": "available", "path": str(sheet), "n": len(rows),
            "skill_sha256": hashlib.sha256((source / "best_skill.md").read_bytes()).hexdigest(),
            "em": sum(r["hard"] for r in rows) / len(rows),
            "f1": sum(r["soft"] for r in rows) / len(rows)}


def evaluate_run(run: Path):
    run = run.resolve()
    meta = read_json(run / "manifest.json")
    if meta["benchmark"] != "triviaqa":
        raise ValueError("This evaluator requires a TriviaQA evolution run")
    dest = run / "test"
    dest.mkdir(mode=0o700, exist_ok=True)
    with (dest / ".evaluator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"A test evaluator is already attached to {run}") from None
        status = dest / "status.json"
        try:
            save_json(status, {"status": "waiting_for_evolution", "time": now()})
            evolution, skill = wait_for_winner(run)
            source = Path(meta["source"])
            cfg = read_json(source / "config.json")
            if fingerprint(cfg) != meta["config_sha256"]:
                raise ValueError("Source adapter config differs from the evolution manifest fingerprint")
            # Recover adapter-specific options. Stage two is authoritative for the
            # actual target model, backend, worker count and output token cap.
            cfg.update(meta["source_config"])
            split = Path(meta["split_path"]).parent.parent / "test" / "items.json"
            items = read_json(split)
            if len(items) != 1000 or len({str(q["id"]) for q in items}) != len(items):
                raise ValueError("Expected 1,000 unique TriviaQA test IDs")
            identity = {
                "evolution_run": str(run), "evolution_revision": meta["git_revision"],
                "best": evolution["best"], "benchmark": meta["benchmark"], "test_split": str(split),
                "test_sha256": fingerprint(items), "test_size": len(items),
                "target_model": cfg["target_model"], "target_backend": cfg["target_backend"],
                "workers": cfg["workers"], "max_completion_tokens": cfg["max_completion_tokens"],
                "source_config_sha256": meta["config_sha256"],
            }
            signature = dest / "manifest.json"
            if signature.exists():
                previous = read_json(signature)
                if any(previous.get(k) != v for k, v in identity.items()):
                    raise ValueError(f"Existing test inputs do not match the frozen selection: {signature}")
            else:
                revision = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]),
                                                    "rev-parse", "HEAD"], text=True).strip()
                save_json(signature, dict(identity, created_utc=now(), evaluator_revision=revision))
            frozen = dest / "best_skill.md"
            if frozen.exists() and frozen.read_bytes() != skill.encode():
                raise ValueError("Previously frozen test skill has changed")
            if not frozen.exists():
                tmp = dest / "best_skill.tmp"
                tmp.write_bytes(skill.encode())
                tmp.replace(frozen)
            final = dest / "summary.json"
            if final.exists():
                existing = read_json(final)
                if existing.get("status") == "complete":
                    digest = hashlib.sha256((dest / "results.jsonl").read_bytes()).hexdigest()
                    if digest != existing["results_sha256"]:
                        raise ValueError("Saved test answers do not match the completed report")
                    save_json(status, {"status": "complete", "summary": str(final), "time": now()})
                    print(f"Already complete: {final}", flush=True)
                    return existing
            reference = first_stage_reference(source, items)
            print(f"Testing {run.name}: {evolution['best']['id']}, {len(items)} items, "
                  f"{cfg['target_model']}, workers={cfg['workers']}", flush=True)
            save_json(status, {"status": "testing", "time": now(), "skill_sha256": identity["best"]["sha256"]})
            metrics = run_test(adapter=archived_adapter(cfg), items=items, skill=skill, out_dir=dest)
            digest = hashlib.sha256((dest / "results.jsonl").read_bytes()).hexdigest()
            result = dict(metrics, results_sha256=digest, finished_utc=now(), best=identity["best"],
                          target_model=cfg["target_model"], first_stage_reference=reference)
            if reference["status"] == "available":
                result.update(skill_changed=identity["best"]["sha256"] != reference["skill_sha256"],
                              delta_em_pp=100 * (metrics["em"] - reference["em"]),
                              delta_f1_pp=100 * (metrics["f1"] - reference["f1"]))
            save_json(final, result)
            if result["status"] != "complete":
                raise RuntimeError(f"{result['n_failed']} test items still failed; rerun to retry: {final}")
            save_json(status, {"status": "complete", "summary": str(final), "time": now()})
            print(f"Final test EM={metrics['em']:.4f}, F1={metrics['f1']:.4f}; {final}", flush=True)
            return result
        except Exception as error:
            save_json(status, {"status": "failed", "type": type(error).__name__, "time": now()})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution-dir", required=True, type=Path, help="exact second-stage output directory")
    args = parser.parse_args(argv)
    evaluate_run(args.evolution_dir)


if __name__ == "__main__":
    main()
