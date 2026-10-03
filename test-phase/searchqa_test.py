#!/usr/bin/env python3
"""Select SearchQA skills by validation-set coverage and evaluate them on test.

This script deliberately keeps selection data and test data separate: candidate
selection reads only the saved ``valid_seen`` results, while model rollouts are
performed only after the five candidates have been selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "outputs" / (
    "skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    skill_path: Path
    results_path: Path
    correct_ids: frozenset[str]
    accuracy: float
    version: int


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(row)
    return rows


def version_of(name: str) -> int:
    match = re.match(r"skill_v(\d+)", name)
    return int(match.group(1)) if match else 10**9


def candidate_sources(root: Path) -> list[tuple[str, Path, Path]]:
    sources: list[tuple[str, Path, Path]] = []
    baseline = root / "selection_eval_baseline" / "results.jsonl"
    baseline_skill = root / "skills" / "skill_v0000.md"
    if baseline.is_file() and baseline_skill.is_file():
        sources.append(("skill_v0000", baseline_skill, baseline))

    for results in sorted((root / "steps").glob("step_*/selection_eval/results.jsonl")):
        match = re.fullmatch(r"step_(\d+)", results.parent.parent.name)
        if not match:
            continue
        step = int(match.group(1))
        skill = results.parent.parent / "candidate_skill.md"
        if skill.is_file():
            sources.append((f"skill_v{step:04d}", skill, results))

    # The final skill is a normal candidate.  Its validation result is still
    # valid_seen (final_selection_eval), never the valid_unseen test result.
    final_results = root / "final_selection_eval" / "results.jsonl"
    final_skill = root / "skills" / "skill_v0040.md"
    if final_results.is_file() and final_skill.is_file():
        # Keep the final snapshot distinct from step 40's candidate when both
        # exist; they can have different contents and different validation
        # results even though both originate around version 40.
        sources.append(("skill_v0040_final", final_skill, final_results))
    return sources


def load_candidates(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_hashes: dict[str, str] = {}
    for name, skill_path, results_path in candidate_sources(root):
        content_hash = hashlib.sha256(skill_path.read_bytes()).hexdigest()
        if content_hash in seen_hashes:
            log(f"[select] duplicate {name}; representative={seen_hashes[content_hash]}")
            continue
        rows = load_jsonl(results_path)
        if not rows:
            log(f"[select] skip {name}: empty results")
            continue
        ids = {str(row["id"]) for row in rows if int(row.get("hard", 0)) == 1}
        seen_hashes[content_hash] = name
        candidates.append(Candidate(name, skill_path, results_path, frozenset(ids), len(ids) / len(rows), version_of(name)))
        log(f"[select] loaded {name}: n={len(rows)} hard={len(ids)} acc={len(ids)/len(rows):.4f}")
    if not candidates:
        raise RuntimeError(f"no usable candidates found under {root}")
    return candidates


def select_cover(candidates: list[Candidate], vote_num: int) -> list[Candidate]:
    if vote_num <= 0 or vote_num > len(candidates):
        raise ValueError(f"cover count={vote_num} but candidates={len(candidates)}")
    remaining = set().union(*(c.correct_ids for c in candidates))
    pool = list(candidates)
    selected: list[Candidate] = []
    for index in range(1, vote_num + 1):
        chosen = max(pool, key=lambda c: (len(c.correct_ids & remaining), len(c.correct_ids), -c.version))
        gain = len(chosen.correct_ids & remaining)
        selected.append(chosen)
        pool.remove(chosen)
        remaining.difference_update(chosen.correct_ids)
        log(f"[select] round={index}/{vote_num} skill={chosen.name} gain={gain} remaining={len(remaining)}")
    return selected


def configure_target(cfg: dict) -> None:
    from skillopt.model import (
        configure_openai_compatible,
        set_reasoning_effort,
        set_target_backend,
        set_target_deployment,
    )

    backend = cfg.get("target_backend", "openai_compatible")
    set_target_backend(backend)
    set_target_deployment(cfg.get("target_model", ""))
    set_reasoning_effort(cfg.get("reasoning_effort") or None)
    if backend == "openai_compatible":
        configure_openai_compatible(
            target_model=cfg.get("target_model") or None,
            target_base_url=os.getenv("TARGET_OPENAI_COMPATIBLE_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
            target_api_key=os.getenv("TARGET_OPENAI_COMPATIBLE_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY"),
            max_tokens=cfg.get("max_completion_tokens"),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--vote-num", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    root = args.result_root.resolve()
    with (root / "config.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    candidates = load_candidates(root)
    selected = select_cover(candidates, args.vote_num)

    model_name = str(cfg.get("target_model", "unknown")).replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = REPO_ROOT / "outputs" / f"searchqa_{model_name}_{timestamp}_cover"
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump({"mode": "cover", "source_result_root": str(root), "selected": [asdict(c) | {"skill_path": str(c.skill_path), "results_path": str(c.results_path), "correct_ids": sorted(c.correct_ids)} for c in selected]}, handle, ensure_ascii=False, indent=2)

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.eval_only import get_adapter
    from skillopt.utils import compute_score

    configure_target(cfg)
    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    test_items = adapter.build_eval_env(0, "valid_unseen", cfg.get("seed", 42))
    log(f"[test] split=valid_unseen items={len(test_items)} output={output_root}")

    all_results: dict[str, list[dict]] = {}
    for rank, candidate in enumerate(selected, 1):
        skill_out = output_root / f"{rank:03d}_{candidate.name}"
        skill_content = candidate.skill_path.read_text(encoding="utf-8")
        # SearchQAAdapter takes its worker count from the saved run config;
        # keeping that path reuses the project's existing rollout behavior.
        results = adapter.rollout(test_items, skill_content, str(skill_out))
        hard, soft = compute_score(results)
        all_results[candidate.name] = results
        with (skill_out / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump({"skill": candidate.name, "split": "valid_unseen", "n": len(results), "hard": hard, "soft": soft}, handle, ensure_ascii=False, indent=2)
        log(f"[test] {candidate.name}: hard={hard:.4f} soft={soft:.4f}")

    ids = set().union(*(str(r["id"]) for rows in all_results.values() for r in rows))
    covered = {str(r["id"]) for rows in all_results.values() for r in rows if int(r.get("hard", 0)) == 1}
    summary = {"mode": "cover", "split": "valid_unseen", "skills": list(all_results), "items": len(ids), "covered_by_at_least_one": len(covered), "coverage": len(covered) / max(len(ids), 1)}
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    log(f"[done] coverage={summary['coverage']:.4f} saved={output_root}")


if __name__ == "__main__":
    main()
