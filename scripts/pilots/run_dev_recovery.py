"""Standalone paired dev-only pilot. Run from the repository root with -m."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import subprocess
from pathlib import Path

from skillopt.engine.trainer import _load_meta_skill_content
from skillopt.envs.base import EnvAdapter
from skillopt.envs.searchqa.adapter import SearchQAAdapter
from skillopt.model import set_backend, set_optimizer_deployment, set_reasoning_effort, set_target_deployment

from .dev_recovery_archive import build_dev_archive
from .dev_recovery_eval import evaluate_candidates
from .dev_recovery_generation import generate_pair


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=["hotpotqa", "triviaqa"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    source = root / "outputs" / (
        f"skillopt_{args.bench}_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075"
    )
    out = args.out.resolve()
    if not out.is_relative_to(root / "outputs" / "dev_recovery_pilots"):
        raise ValueError("Use a dedicated directory inside outputs/dev_recovery_pilots")
    if args.resume:
        if not (out / "manifest.json").is_file():
            raise ValueError("Resume requires a pilot manifest")
    else:
        out.mkdir(parents=True, exist_ok=False)

    cfg = read(source / "config.json")
    if not cfg.get("selection_early_stop") or cfg.get("train_early_stop_rho") != 0.75:
        raise ValueError("Archive does not match the requested early-stop configuration")
    split = root / cfg["split_dir"]
    if split != root / "data" / f"{args.bench}_split":
        raise ValueError(f"Unexpected archived split: {split}")
    train = read(split / "train" / "items.json")
    dev = read(split / "val" / "items.json")
    dev_ids = [str(q["id"]) for q in dev]
    if len(train) != 1600 or len(dev_ids) != 200 or len(set(dev_ids)) != 200:
        raise ValueError("Unexpected split size or duplicate dev IDs")
    if {str(q["id"]) for q in train} & set(dev_ids):
        raise ValueError("Overlapping train/dev IDs")

    hist = read(source / "history.json")
    archive = build_dev_archive(source, dev_ids)
    anchor = archive["best_skill"]
    model = cfg["target_model"]
    for role in ["optimizer", "target"]:
        if cfg[role + "_backend"] != "openai_compatible" or cfg[role + "_model"] != model:
            raise ValueError("Unexpected model in archive")
        if int(os.environ.get(role.upper() + "_OPENAI_COMPATIBLE_MAX_TOKENS", 0)) != 131072:
            raise ValueError("Both model roles require the archived 131072-token cap")
    if not os.getenv("OPENAI_COMPATIBLE_API_KEY"):
        raise ValueError("Missing model API credential")
    set_backend("openai_compatible")
    set_target_deployment(model)
    set_optimizer_deployment(model)
    set_reasoning_effort(cfg["reasoning_effort"])
    # The stock dataset setup loads all splits. Pass the explicit train/val lists instead.
    keys = inspect.signature(SearchQAAdapter).parameters
    adapter = SearchQAAdapter(**{k: cfg[k] for k in keys if k in cfg})
    EnvAdapter.setup(adapter, cfg)
    batch = cfg["batch_size"]
    seed = cfg["seed"] + 1000 * (hist[-1]["epoch"] + 1)
    sampled = random.Random(seed).sample(train, 2 * batch)
    meta = _load_meta_skill_content(str(source), hist[-1]["epoch"] - 1)
    manifest = {
        "benchmark": args.bench, "source": str(source), "anchor_step": archive["best_step"],
        "anchor_em": archive["best_hard"], "anchor_sha256": hashlib.sha256(anchor.encode()).hexdigest(),
        "model": model, "workers": cfg["workers"], "batch_size": batch,
        "dev_screen_batch": cfg["selection_early_stop_batch"], "role_token_caps": [131072, 131072],
        "edit_budget": hist[-1]["edit_budget"], "sampling_seed": seed,
        "train_pair_ids": [[row["id"] for row in sampled[j:j+batch]] for j in (0, batch)],
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    manifest_path = out / "manifest.json"
    if args.resume and read(manifest_path) != manifest:
        raise ValueError("Pilot inputs or code revision changed")
    save(manifest_path, manifest)
    save(out / "archive_evidence.json", archive["evidence"])
    print(f"{args.bench}: anchor v{archive['best_step']}, EM={archive['best_hard']}, "
          f"paired train batches of {batch}, dev N={len(dev)}", flush=True)
    candidates = {"ordinary": [], "recovery": []}
    for j in range(2):
        result = generate_pair(adapter=adapter, cfg=cfg, anchor=anchor,
                               train_items=sampled[j*batch:(j+1)*batch], seed=seed+j,
                               out_dir=out / f"pair_{j+1}", edit_budget=manifest["edit_budget"],
                               meta_context=meta, step_context="", donor_bank=archive["evidence"])
        for arm in candidates:
            candidates[arm].append(result[arm])
    failures = []
    for arm, values in candidates.items():
        if len({s.strip() for s in [anchor, *values]}) != 3:
            failures.append(arm)
    if failures:
        print("Duplicate or unchanged candidates: " + ", ".join(failures), flush=True)
    print("Freezing candidate texts and starting dev screening", flush=True)
    evaluate_candidates(adapter=adapter, dev_items=dev, history=archive["history"],
                        anchor_skill=anchor, anchor_rows=archive["best_rows"],
                        candidates=candidates, out_dir=out / "dev",
                        evidence_ids=archive["evidence_ids"])
    print("Completed dev-only pilot", flush=True)


if __name__ == "__main__":
    main()
