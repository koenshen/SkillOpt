#!/usr/bin/env python3
"""SearchQA skill selection and evaluation.

The candidate pool is built only from complete ``valid_seen`` evaluations.
The current implementation provides the greedy ``cover`` selector; future
selectors can consume the same Candidate objects and output protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / (
    "skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315"
)
VALIDATION_IDS_PATH = REPO_ROOT / "data" / "searchqa_split" / "val" / "items.json"


@dataclass(frozen=True)
class Candidate:
    name: str
    skill_path: Path
    validation_results_path: Path
    records_by_id: dict[str, dict]
    correct_ids: frozenset[str]
    version: int

    @property
    def accuracy(self) -> float:
        return len(self.correct_ids) / max(len(self.records_by_id), 1)


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def load_expected_validation_ids() -> frozenset[str]:
    with VALIDATION_IDS_PATH.open(encoding="utf-8") as handle:
        items = json.load(handle)
    ids = {str(item["id"]) for item in items}
    if not ids:
        raise RuntimeError(f"empty validation split: {VALIDATION_IDS_PATH}")
    return frozenset(ids)


def version_of(name: str) -> int:
    match = re.match(r"skill_v(\d+)", name)
    return int(match.group(1)) if match else 10**9


def final_skill_path(root: Path) -> Path | None:
    state_path = root / "runtime_state.json"
    if state_path.is_file():
        with state_path.open(encoding="utf-8") as handle:
            raw_path = json.load(handle).get("current_skill_path")
        if raw_path:
            path = Path(raw_path)
            if path.is_file():
                return path
    return None


def candidate_sources(root: Path) -> list[tuple[str, Path, Path]]:
    """Discover source pairs; selection later enforces complete valid_seen IDs."""
    sources: list[tuple[str, Path, Path]] = []

    baseline_results = root / "selection_eval_baseline" / "results.jsonl"
    baseline_skill = root / "skills" / "skill_v0000.md"
    if baseline_results.is_file() and baseline_skill.is_file():
        sources.append(("skill_v0000", baseline_skill, baseline_results))

    for results_path in sorted((root / "steps").glob("step_*/selection_eval/results.jsonl")):
        match = re.fullmatch(r"step_(\d+)", results_path.parent.parent.name)
        if not match:
            continue
        step = int(match.group(1))
        skill_path = results_path.parent.parent / "candidate_skill.md"
        if skill_path.is_file():
            sources.append((f"skill_v{step:04d}_step_candidate", skill_path, results_path))

    final_results = root / "final_selection_eval" / "results.jsonl"
    current_path = final_skill_path(root)
    if final_results.is_file() and current_path is not None:
        sources.append((f"skill_v{version_of(current_path.stem):04d}_final", current_path, final_results))
    return sources


def load_candidates(root: Path) -> list[Candidate]:
    expected_ids = load_expected_validation_ids()
    candidates: list[Candidate] = []
    seen_hashes: dict[str, str] = {}

    for name, skill_path, results_path in candidate_sources(root):
        content_hash = hashlib.sha256(skill_path.read_bytes()).hexdigest()
        if content_hash in seen_hashes:
            log(f"[select] duplicate {name}; representative={seen_hashes[content_hash]}")
            continue

        rows = load_jsonl(results_path)
        records_by_id = {str(row.get("id")): row for row in rows if row.get("id") is not None}
        result_ids = frozenset(records_by_id)
        if len(rows) != len(records_by_id) or result_ids != expected_ids:
            log(
                f"[select] skip {name}: incomplete validation results "
                f"rows={len(rows)} unique_ids={len(result_ids)} expected={len(expected_ids)}"
            )
            continue

        seen_hashes[content_hash] = name
        correct_ids = frozenset(
            question_id
            for question_id, row in records_by_id.items()
            if int(row.get("hard", 0)) == 1
        )
        candidate = Candidate(
            name=name,
            skill_path=skill_path,
            validation_results_path=results_path,
            records_by_id=records_by_id,
            correct_ids=correct_ids,
            version=version_of(name),
        )
        candidates.append(candidate)
        log(
            f"[select] loaded {name}: n={len(records_by_id)} "
            f"hard={len(correct_ids)} acc={candidate.accuracy:.4f}"
        )

    if not candidates:
        raise RuntimeError(f"no complete valid_seen candidates found under {root}")
    return candidates


def select_cover(candidates: list[Candidate], num_skills: int) -> list[Candidate]:
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")

    remaining = set().union(*(candidate.correct_ids for candidate in candidates))
    pool = list(candidates)
    selected: list[Candidate] = []
    for round_index in range(1, num_skills + 1):
        chosen = max(
            pool,
            key=lambda candidate: (
                len(candidate.correct_ids & remaining),
                len(candidate.correct_ids),
                -candidate.version,
            ),
        )
        gain = len(chosen.correct_ids & remaining)
        selected.append(chosen)
        pool.remove(chosen)
        remaining.difference_update(chosen.correct_ids)
        log(
            f"[select] round={round_index}/{num_skills} skill={chosen.name} "
            f"gain={gain} remaining={len(remaining)}"
        )
    return selected


def select_best_repeat(candidates: list[Candidate], num_skills: int) -> list[Candidate]:
    """Return the valid_seen-best skill repeated ``num_skills`` times."""
    if not candidates:
        raise ValueError("cannot select from an empty candidate pool")
    if num_skills <= 0:
        raise ValueError(f"num_skills={num_skills}; expected a positive value")

    best = max(
        candidates,
        key=lambda candidate: (len(candidate.correct_ids), -candidate.version),
    )
    log(
        f"[select] best_repeat best={best.name} "
        f"hard={len(best.correct_ids)} acc={best.accuracy:.4f} "
        f"repetitions={num_skills}"
    )
    return [best] * num_skills


def select_top_k(candidates: list[Candidate], num_skills: int) -> list[Candidate]:
    """Return the top ``num_skills`` distinct candidates by valid_seen score."""
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")

    ranked = sorted(
        candidates,
        key=lambda candidate: (-len(candidate.correct_ids), candidate.version),
    )
    selected = ranked[:num_skills]
    for rank, candidate in enumerate(selected, 1):
        log(
            f"[select] top_k rank={rank}/{num_skills} skill={candidate.name} "
            f"hard={len(candidate.correct_ids)} acc={candidate.accuracy:.4f}"
        )
    return selected


def configure_runtime(cfg: dict) -> None:
    """Configure the existing SkillOpt target runtime for this evaluation."""
    from skillopt.model import (
        configure_openai_compatible,
        set_reasoning_effort,
        set_target_backend,
        set_target_deployment,
    )

    backend = cfg.get("target_backend", "openai_compatible")
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
    model = os.environ.get("OPENAI_COMPATIBLE_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("OPENAI_COMPATIBLE_BASE_URL", base_url),
            ("OPENAI_COMPATIBLE_API_KEY", api_key),
            ("OPENAI_COMPATIBLE_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "missing required environment variable(s): " + ", ".join(missing)
        )
    if backend != "openai_compatible":
        raise RuntimeError(
            f"config target_backend={backend!r}; expected 'openai_compatible'"
        )

    set_target_backend(backend)
    set_target_deployment(model)
    set_reasoning_effort(cfg.get("reasoning_effort") or None)
    configure_openai_compatible(
        target_base_url=base_url,
        target_api_key=api_key,
        target_model=model,
        max_tokens=cfg.get("max_completion_tokens"),
    )


def candidate_metadata(candidate: Candidate) -> dict:
    return {
        "name": candidate.name,
        "skill_path": str(candidate.skill_path),
        "validation_results_path": str(candidate.validation_results_path),
        "version": candidate.version,
        "validation_n": len(candidate.records_by_id),
        "validation_hard": len(candidate.correct_ids),
        "validation_hard_accuracy": candidate.accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("cover", "best_repeat", "top_k"),
        default="cover",
    )
    parser.add_argument("--num-skills", type=int, required=True)
    args = parser.parse_args()

    root = RESULT_ROOT.resolve()
    with (root / "config.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    candidates = load_candidates(root)
    selectors = {
        "cover": select_cover,
        "best_repeat": select_best_repeat,
        "top_k": select_top_k,
    }
    selected = selectors[args.mode](candidates, args.num_skills)
    sys.path.insert(0, str(REPO_ROOT))
    configure_runtime(cfg)

    model_name = os.environ["OPENAI_COMPATIBLE_MODEL"].replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = REPO_ROOT / "outputs" / (
        f"skillopt_searchqa_{model_name}_{timestamp}_{args.mode}_numskills{args.num_skills}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_method": args.mode,
                "source_result_root": str(root),
                "candidate_count": len(candidates),
                "selected": [candidate_metadata(candidate) for candidate in selected],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    from scripts.eval_only import get_adapter
    from skillopt.utils import compute_score

    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    test_items = adapter.build_eval_env(0, "valid_unseen", cfg.get("seed", 42))
    test_ids = {str(item["id"]) for item in test_items}
    log(f"[test] split=valid_unseen items={len(test_items)} output={output_root}")

    per_skill: list[dict] = []
    test_correct_union: set[str] = set()
    for rank, candidate in enumerate(selected, 1):
        skill_out = output_root / f"{rank:03d}_{candidate.name}"
        skill_content = candidate.skill_path.read_text(encoding="utf-8")
        results = adapter.rollout(test_items, skill_content, str(skill_out))
        result_ids = {str(row["id"]) for row in results if row.get("id") is not None}
        missing_ids = sorted(test_ids - result_ids)
        extra_ids = sorted(result_ids - test_ids)
        hard, soft = compute_score(results)
        test_correct_union.update(
            str(row["id"])
            for row in results
            if row.get("id") is not None and int(row.get("hard", 0)) == 1
        )
        skill_summary = {
            "skill": candidate.name,
            "split": "valid_unseen",
            "n_expected": len(test_ids),
            "n_results": len(results),
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "hard": hard,
            "soft": soft,
        }
        with (skill_out / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(skill_summary, handle, ensure_ascii=False, indent=2)
        per_skill.append(skill_summary)
        log(f"[test] {candidate.name}: hard={hard:.4f} soft={soft:.4f}")

    summary = {
        "selection_method": args.mode,
        "split": "valid_unseen",
        "n_test_items": len(test_ids),
        "selected_skills": [candidate.name for candidate in selected],
        "per_skill": per_skill,
        "covered_by_at_least_one": len(test_correct_union & test_ids),
        "coverage": len(test_correct_union & test_ids) / max(len(test_ids), 1),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    log(f"[done] coverage={summary['coverage']:.4f} saved={output_root}")


if __name__ == "__main__":
    main()
