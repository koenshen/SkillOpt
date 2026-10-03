#!/usr/bin/env python3
"""SearchQA skill selection and evaluation.

The candidate pool is built only from complete ``valid_seen`` evaluations.
Selectors include greedy ``cover``, repeated-best, top-k, and a hybrid mode
that optimises the final skill using the shared voting protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PHASE_ROOT = Path(__file__).resolve().parent
if str(TEST_PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_PHASE_ROOT))

from test_vote_result import (  # noqa: E402
    DatasetHandler,
    SkillResults,
    build_vote_row,
    get_gold_answers,
    get_dataset_handler,
    is_valid_vote,
)


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


def select_max_cover_milp(
    candidates: list[Candidate],
    num_skills: int,
    initially_covered_ids: frozenset[str] = frozenset(),
) -> list[Candidate]:
    """Select an exact maximum-coverage set with a binary MILP.

    ``x[j]`` indicates whether candidate ``j`` is selected and ``y[q]``
    indicates whether question ``q`` is covered by at least one selected
    candidate.  ``initially_covered_ids`` allows a fixed prefix, such as
    ``best_skill``, to contribute to the union without being re-selected.
    The first solve maximises ``sum(y)`` over the residual questions. A
    second solve keeps that optimum fixed and minimises a candidate-order
    score to make the choice among equivalent optima stable where possible.
    """
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError(
            "max_cover_milp requires scipy>=1.9 and numpy in the active SkillOpt environment"
        ) from exc

    def report(message: str) -> None:
        log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [max_cover_milp] {message}")

    ordered = sorted(candidates, key=lambda candidate: (candidate.version, candidate.name))
    all_covered_ids = set().union(*(candidate.correct_ids for candidate in ordered))
    covered_question_ids = sorted(all_covered_ids - set(initially_covered_ids))
    n_candidates = len(ordered)
    n_questions = len(covered_question_ids)
    question_index = {question_id: index for index, question_id in enumerate(covered_question_ids)}
    variable_count = n_candidates + n_questions

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coeffs: dict[int, float], lo=-np.inf, hi=np.inf) -> None:
        row_index = len(lower)
        for column, value in coeffs.items():
            if value:
                row_indices.append(row_index)
                col_indices.append(column)
                values.append(float(value))
        lower.append(float(lo))
        upper.append(float(hi))

    # Select exactly num_skills candidates.
    add_constraint({index: 1 for index in range(n_candidates)}, num_skills, num_skills)
    # y[q] can be one only if at least one selected skill covers q.
    for question_id, question_offset in question_index.items():
        covering = [
            candidate_index
            for candidate_index, candidate in enumerate(ordered)
            if question_id in candidate.correct_ids
        ]
        coefficients = {n_candidates + question_offset: 1}
        coefficients.update({candidate_index: -1 for candidate_index in covering})
        add_constraint(coefficients, hi=0)

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(lower), variable_count),
    ).tocsc()
    base_constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    integrality = np.ones(variable_count)

    coverage_objective = np.zeros(variable_count)
    coverage_objective[n_candidates:] = -1.0  # scipy.milp minimises
    report(
        f"model variables={variable_count} ({n_candidates} skills, "
        f"{n_questions} residual questions) constraints={len(lower)} "
        f"num_skills={num_skills} initially_covered={len(initially_covered_ids)}"
    )

    started = time.monotonic()
    report("solve stage=coverage started; HiGHS progress below uses minimisation signs")
    coverage_result = milp(
        c=coverage_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=base_constraints,
        options={"disp": True, "mip_rel_gap": 0.0},
    )
    coverage_bound = getattr(coverage_result, "mip_dual_bound", None)
    report(
        f"solve stage=coverage finished status={coverage_result.status} "
        f"success={coverage_result.success} elapsed={time.monotonic() - started:.1f}s "
        f"objective={-coverage_result.fun if coverage_result.fun is not None else None} "
        f"best_bound={-coverage_bound if coverage_bound is not None else None} "
        f"mip_gap={getattr(coverage_result, 'mip_gap', None)} message={coverage_result.message}"
    )
    if (
        not coverage_result.success
        or coverage_result.x is None
        or coverage_result.fun is None
        or coverage_result.status != 0
    ):
        raise RuntimeError(
            f"maximum-coverage MILP did not prove optimality: {coverage_result.message}"
        )
    optimum = int(round(-float(coverage_result.fun)))

    # Fix the exact coverage optimum and choose a deterministic candidate set.
    # Prefer earlier candidates in the deterministic ``ordered`` list without
    # changing the primary coverage. Equal secondary scores can still exist;
    # they have identical maximum coverage.
    tie_break_weights = np.zeros(variable_count)
    tie_break_weights[:n_candidates] = np.arange(1, n_candidates + 1, dtype=float)
    tie_constraints = [
        base_constraints,
        LinearConstraint(
            np.asarray(coverage_objective).reshape(1, -1), -optimum, -optimum
        ),
    ]
    started = time.monotonic()
    report(f"solve stage=tie_break started coverage_fixed={optimum}")
    tie_result = milp(
        c=tie_break_weights,
        integrality=integrality,
        bounds=bounds,
        constraints=tie_constraints,
        options={"disp": True, "mip_rel_gap": 0.0},
    )
    tie_bound = getattr(tie_result, "mip_dual_bound", None)
    report(
        f"solve stage=tie_break finished status={tie_result.status} "
        f"success={tie_result.success} elapsed={time.monotonic() - started:.1f}s "
        f"objective={tie_result.fun if tie_result.fun is not None else None} "
        f"best_bound={tie_bound} mip_gap={getattr(tie_result, 'mip_gap', None)} "
        f"message={tie_result.message}"
    )
    if not tie_result.success or tie_result.x is None or tie_result.status != 0:
        raise RuntimeError(
            f"maximum-coverage tie-break MILP did not prove optimality: {tie_result.message}"
        )

    selected = [
        candidate for index, candidate in enumerate(ordered) if tie_result.x[index] > 0.5
    ]
    if len(selected) != num_skills:
        raise RuntimeError(f"MILP selected {len(selected)} skills; expected {num_skills}")
    selected_covered_ids = set().union(*(candidate.correct_ids for candidate in selected))
    verified_residual_coverage = len(
        selected_covered_ids - set(initially_covered_ids)
    )
    if verified_residual_coverage != optimum:
        raise RuntimeError(
            f"maximum-coverage verification mismatch: solver={optimum} "
            f"selected_residual_union={verified_residual_coverage}"
        )
    verified_coverage = len(set(initially_covered_ids) | selected_covered_ids)
    report(
        f"verified selected={[candidate.name for candidate in selected]} "
        f"coverage={verified_coverage}/{len(ordered[0].records_by_id)}"
    )
    return selected


def select_best_max_cover_milp(
    candidates: list[Candidate], num_skills: int, root: Path
) -> list[Candidate]:
    """Fix the last complete ``best_skill`` result, then maximise coverage.

    The existing maximum-coverage MILP chooses only the remaining
    ``num_skills - 1`` candidates, while the fixed best skill contributes its
    already-covered question IDs to the objective through
    ``initially_covered_ids``.
    """
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")

    best = _load_last_best_skill(root, candidates)
    best_hash = hashlib.sha256(best.skill_path.read_bytes()).hexdigest()
    remaining = [
        candidate
        for candidate in candidates
        if hashlib.sha256(candidate.skill_path.read_bytes()).hexdigest() != best_hash
    ]
    if num_skills - 1 > len(remaining):
        raise ValueError(
            f"num_skills={num_skills} requires {num_skills - 1} remaining skills, "
            f"but only {len(remaining)} are available"
        )
    if num_skills == 1:
        log(f"[select] best_max_cover_milp selected={[best.name]}")
        return [best]

    selected_remaining = select_max_cover_milp(
        remaining,
        num_skills - 1,
        initially_covered_ids=best.correct_ids,
    )
    selected = [best, *selected_remaining]
    coverage = len(set().union(*(candidate.correct_ids for candidate in selected)))
    log(
        f"[select] best_max_cover_milp selected={[candidate.name for candidate in selected]} "
        f"coverage={coverage}/{len(best.records_by_id)}"
    )
    return selected


def _candidate_for_voting(candidate: Candidate, rank: int) -> SkillResults:
    """Adapt a SearchQA validation candidate to the shared voting protocol."""
    return SkillResults(
        rank=rank,
        name=candidate.name,
        path=candidate.skill_path,
        rows_by_id=candidate.records_by_id,
    )


def _validation_vote_score(
    selected: list[Candidate],
    handler: DatasetHandler,
) -> tuple[int, float]:
    """Score a selected validation set using the canonical voting logic."""
    vote_skills = [
        _candidate_for_voting(candidate, rank)
        for rank, candidate in enumerate(selected, 1)
    ]
    question_ids = sorted(vote_skills[0].rows_by_id)
    vote_rows = [
        build_vote_row(question_id, vote_skills, handler)
        for question_id in question_ids
    ]
    hard_total = sum(int(row["hard"]) for row in vote_rows)
    soft_total = sum(float(row["soft"]) for row in vote_rows)
    return hard_total, soft_total


def _complete_records(
    results_path: Path, expected_ids: frozenset[str]
) -> dict[str, dict] | None:
    """Load a complete validation result file, or return None if incomplete."""
    rows = load_jsonl(results_path)
    records = {str(row["id"]): row for row in rows if row.get("id") is not None}
    if len(records) != len(rows) or frozenset(records) != expected_ids:
        return None
    return records


def _load_last_best_skill(root: Path, candidates: list[Candidate]) -> Candidate:
    """Use the last complete validation run matching ``best_skill.md``."""
    best_path = root / "best_skill.md"
    if not best_path.is_file():
        raise FileNotFoundError(f"best skill file not found: {best_path}")

    expected_ids = load_expected_validation_ids()
    best_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()
    matches: list[tuple[int, str, Path, dict[str, dict]]] = []
    for source_index, (name, skill_path, results_path) in enumerate(candidate_sources(root)):
        if hashlib.sha256(skill_path.read_bytes()).hexdigest() != best_hash:
            continue
        records = _complete_records(results_path, expected_ids)
        if records is not None:
            matches.append((source_index, name, results_path, records))

    if not matches:
        raise RuntimeError(
            f"no complete valid_seen result matches best skill: {best_path}"
        )

    _, source_name, results_path, records = matches[-1]
    correct_ids = frozenset(
        question_id
        for question_id, row in records.items()
        if int(row.get("hard", 0) or 0) == 1
    )
    version = version_of(source_name)
    log(
        f"[select] best_skill={best_path} matched_runs={len(matches)} "
        f"using_last={source_name} results={results_path} "
        f"hard={len(correct_ids)}/{len(records)} "
        f"acc={len(correct_ids) / max(len(records), 1):.4f}"
    )
    return Candidate(
        name="best_skill",
        skill_path=best_path,
        validation_results_path=results_path,
        records_by_id=records,
        correct_ids=correct_ids,
        version=version,
    )


def _select_vote_combination(
    prefix: list[Candidate],
    remaining: list[Candidate],
    choose_count: int,
    label: str,
) -> list[Candidate]:
    """Exhaustively choose a combination using the canonical vote scorer."""
    if choose_count < 0 or choose_count > len(remaining):
        raise ValueError(
            f"choose_count={choose_count} but remaining={len(remaining)}"
        )

    ordered = sorted(remaining, key=lambda candidate: (candidate.version, candidate.name))
    handler = get_dataset_handler("searchqa")
    total = math.comb(len(ordered), choose_count)
    started = time.monotonic()
    best_key: tuple[int, float] | None = None
    best_combo: tuple[Candidate, ...] | None = None
    question_count = len(prefix[0].records_by_id) if prefix else len(ordered[0].records_by_id)

    for completed, combo in enumerate(itertools.combinations(ordered, choose_count), 1):
        hard_total, soft_total = _validation_vote_score([*prefix, *combo], handler)
        key = (hard_total, soft_total)
        if best_key is None or key > best_key:
            best_key = key
            best_combo = combo

        if completed % 10 == 0 or completed == total:
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (total - completed) / rate if rate > 0 else 0.0
            best_hard = f"{best_key[0]}/{question_count}" if best_key else "-"
            best_soft = best_key[1] / max(question_count, 1) if best_key else 0.0
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log(
                f"[{timestamp}] [select] {label} "
                f"{completed}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s "
                f"best_hard={best_hard} best_soft={best_soft:.4f}"
            )

    assert best_combo is not None and best_key is not None
    selected = [*prefix, *best_combo]
    log(
        f"[select] {label} selected={[candidate.name for candidate in selected]} "
        f"vote_hard={best_key[0]}/{question_count} "
        f"vote_soft={best_key[1] / max(question_count, 1):.4f}"
    )
    return selected


def select_best_vote_global(
    candidates: list[Candidate], num_skills: int, root: Path
) -> list[Candidate]:
    """Fix best_skill, then exhaustively optimise the remaining skill set."""
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")

    best = _load_last_best_skill(root, candidates)
    best_hash = hashlib.sha256(best.skill_path.read_bytes()).hexdigest()
    remaining = [
        candidate
        for candidate in candidates
        if hashlib.sha256(candidate.skill_path.read_bytes()).hexdigest() != best_hash
    ]
    if num_skills - 1 > len(remaining):
        raise ValueError(
            f"num_skills={num_skills} requires {num_skills - 1} remaining skills, "
            f"but only {len(remaining)} are available"
        )
    return _select_vote_combination(
        prefix=[best],
        remaining=remaining,
        choose_count=num_skills - 1,
        label="best_vote_global",
    )


def select_vote_global(candidates: list[Candidate], num_skills: int) -> list[Candidate]:
    """Exhaustively select ``num_skills`` candidates for max voting score."""
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")
    return _select_vote_combination(
        prefix=[],
        remaining=candidates,
        choose_count=num_skills,
        label="vote_global",
    )


def select_vote_global_milp(
    candidates: list[Candidate], num_skills: int
) -> list[Candidate]:
    """Optimise hard then soft with the exact first-selected-winner rule.

    x[j] selects a skill. For each question, r[j] identifies the earliest
    selected valid response whose normalised answer has the highest count.
    A separate no-vote variable is allowed only when no valid response exists.
    """
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError(
            "vote_global_milp requires scipy>=1.9 and numpy in the active SkillOpt environment"
        ) from exc

    def report(message: str) -> None:
        log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [milp] {message}")

    handler = get_dataset_handler("searchqa")
    ordered = sorted(candidates, key=lambda candidate: (candidate.version, candidate.name))
    question_ids = sorted(ordered[0].records_by_id)
    n_candidates = len(ordered)
    prepared: list[dict[str, tuple[str, str] | None]] = []
    report(f"preprocess 0/{n_candidates}")
    for index, candidate in enumerate(ordered, 1):
        records = {}
        for question_id in question_ids:
            row = candidate.records_by_id[question_id]
            answer = handler.extract_answer(row)
            records[question_id] = (
                (answer, handler.normalize_answer(answer))
                if is_valid_vote(row, answer, handler) else None
            )
        prepared.append(records)
        if index % 10 == 0 or index == n_candidates:
            report(f"preprocess {index}/{n_candidates} questions={len(question_ids)}")

    variable_count = n_candidates
    hard_coeffs: dict[int, float] = {}
    soft_coeffs: dict[int, float] = {}
    winners: list[dict[int, int]] = []
    no_votes: list[int] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coeffs: dict[int, float], lo=-np.inf, hi=np.inf) -> None:
        row_index = len(lower)
        for column, value in coeffs.items():
            if value:
                row_indices.append(row_index)
                col_indices.append(column)
                values.append(float(value))
        lower.append(float(lo))
        upper.append(float(hi))

    add_constraint({j: 1 for j in range(n_candidates)}, num_skills, num_skills)
    big_m = num_skills + 1
    for question_id in question_ids:
        gold = get_gold_answers(ordered[0].records_by_id[question_id])
        groups: dict[str, list[int]] = {}
        for j in range(n_candidates):
            entry = prepared[j][question_id]
            if entry is not None:
                groups.setdefault(entry[1], []).append(j)
        valid_indices = sorted(j for group in groups.values() for j in group)
        representatives: dict[int, int] = {}
        for j in valid_indices:
            representatives[j] = variable_count
            variable_count += 1
            answer = prepared[j][question_id][0]
            evaluation = handler.evaluate_answer(answer, gold)
            hard_coeffs[representatives[j]] = float(evaluation["em"])
            soft_coeffs[representatives[j]] = float(evaluation["f1"])
        no_vote = variable_count
        variable_count += 1
        # Use the same evaluator even for an empty answer.
        empty = handler.evaluate_answer("", gold)
        hard_coeffs[no_vote] = float(empty["em"])
        soft_coeffs[no_vote] = float(empty["f1"])
        winners.append(representatives)
        no_votes.append(no_vote)
        add_constraint({**{r: 1 for r in representatives.values()}, no_vote: 1}, 1, 1)
        # no_vote=1 => every selected response is invalid.
        add_constraint({**{j: 1 for j in valid_indices}, no_vote: num_skills}, hi=num_skills)
        for j, representative in representatives.items():
            add_constraint({representative: 1, j: -1}, hi=0)
            own_key = prepared[j][question_id][1]
            own_group = groups[own_key]
            # r[j]=1 => own answer count >= every other answer count.
            for key, group in groups.items():
                if key == own_key:
                    continue
                diff = {k: 1 for k in own_group}
                diff.update({k: -1 for k in group})
                diff[representative] = -big_m
                add_constraint(diff, lo=-big_m)
            # An earlier selected valid response must have strictly fewer
            # votes than r[j]'s answer. This also rules out a later response
            # representing the same winning answer.
            for k in valid_indices:
                if k >= j:
                    break
                other_group = groups[prepared[k][question_id][1]]
                diff: dict[int, float] = {}
                for t in own_group:
                    diff[t] = diff.get(t, 0) + 1
                for t in other_group:
                    diff[t] = diff.get(t, 0) - 1
                diff[representative] = -big_m
                diff[k] = diff.get(k, 0) - big_m
                add_constraint(diff, lo=1 - 2 * big_m)

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(lower), variable_count),
    ).tocsc()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    integrality = np.ones(variable_count)
    hard_vector = np.zeros(variable_count)
    soft_vector = np.zeros(variable_count)
    for index, value in hard_coeffs.items():
        hard_vector[index] = value
    for index, value in soft_coeffs.items():
        soft_vector[index] = value
    report(f"model variables={variable_count} constraints={len(lower)} num_skills={num_skills}")

    def solve(stage: str, scores, stage_constraints):
        report(f"solve stage={stage} started; HiGHS progress below uses minimisation signs")
        started = time.monotonic()
        result = milp(
            c=-scores, integrality=integrality, bounds=bounds,
            constraints=stage_constraints, options={"disp": True, "mip_rel_gap": 0.0},
        )
        objective = -result.fun if result.fun is not None else None
        bound = getattr(result, "mip_dual_bound", None)
        report(
            f"solve stage={stage} finished status={result.status} "
            f"elapsed={time.monotonic() - started:.1f}s objective={objective} "
            f"best_bound={-bound if bound is not None else None} "
            f"mip_gap={getattr(result, 'mip_gap', None)} message={result.message}"
        )
        if not result.success or result.x is None or result.status != 0:
            raise RuntimeError(f"MILP {stage} stage did not prove optimality: {result.message}")
        return result

    hard_result = solve("hard", hard_vector, constraints)
    hard_optimum = int(round(-float(hard_result.fun)))
    report(f"stage=soft hard_fixed={hard_optimum} constraints={len(lower) + 1}")
    soft_result = solve(
        "soft", soft_vector,
        [constraints, LinearConstraint(hard_vector.reshape(1, -1), hard_optimum, hard_optimum)],
    )
    selected = [candidate for j, candidate in enumerate(ordered) if soft_result.x[j] > 0.5]
    if len(selected) != num_skills:
        raise RuntimeError(f"MILP selected {len(selected)} skills; expected {num_skills}")
    vote_skills = [_candidate_for_voting(candidate, rank) for rank, candidate in enumerate(selected, 1)]
    verified_rows = [build_vote_row(qid, vote_skills, handler) for qid in question_ids]
    # Verify every winner, not just totals that could conceal compensating errors.
    for q, (qid, row) in enumerate(zip(question_ids, verified_rows)):
        active = [j for j, r in winners[q].items() if soft_result.x[r] > 0.5]
        model_answer = prepared[active[0]][qid][0] if active else ""
        if model_answer != row["voted_answer"]:
            raise RuntimeError(f"MILP winner mismatch at id={qid}")
    verified_hard = sum(row["hard"] for row in verified_rows)
    verified_soft = sum(row["soft"] for row in verified_rows)
    if verified_hard != hard_optimum or not math.isclose(
        verified_soft, -float(soft_result.fun), rel_tol=1e-9, abs_tol=1e-7
    ):
        raise RuntimeError(
            f"MILP verification mismatch: solver=({hard_optimum}, {-soft_result.fun}) "
            f"shared_vote=({verified_hard}, {verified_soft})"
        )
    report(
        f"verified selected={[candidate.name for candidate in selected]} "
        f"vote_hard={verified_hard}/{len(question_ids)} "
        f"vote_soft={verified_soft / max(len(question_ids), 1):.4f}"
    )
    return selected


def select_cover_vote_last(
    candidates: list[Candidate], num_skills: int
) -> list[Candidate]:
    """Select a cover prefix, then optimise the final skill for voting.

    The first ``num_skills - 1`` candidates are selected by ``select_cover``.
    Each remaining candidate is then appended in turn and scored on the
    complete ``valid_seen`` split through the same voting implementation used
    by ``test_vote_result.py``.
    """
    if num_skills <= 0 or num_skills > len(candidates):
        raise ValueError(f"num_skills={num_skills} but candidates={len(candidates)}")

    handler = get_dataset_handler("searchqa")
    prefix = select_cover(candidates, num_skills - 1) if num_skills > 1 else []
    prefix_names = {candidate.name for candidate in prefix}
    remaining = [candidate for candidate in candidates if candidate.name not in prefix_names]

    best_candidate: Candidate | None = None
    best_key: tuple[int, float, int, int] | None = None
    for candidate in remaining:
        trial = prefix + [candidate]
        hard_total, soft_total = _validation_vote_score(trial, handler)
        # Hard voting accuracy is the objective. Soft score, candidate hard
        # score, and version only make equal hard scores deterministic.
        key = (
            hard_total,
            soft_total,
            len(candidate.correct_ids),
            -candidate.version,
        )
        log(
            f"[select] cover_vote_last candidate={candidate.name} "
            f"vote_hard={hard_total}/{len(candidate.records_by_id)} "
            f"vote_soft={soft_total / max(len(candidate.records_by_id), 1):.4f}"
        )
        if best_key is None or key > best_key:
            best_key = key
            best_candidate = candidate

    assert best_candidate is not None
    selected = prefix + [best_candidate]
    log(
        f"[select] cover_vote_last selected="
        f"{[candidate.name for candidate in selected]}"
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
        choices=(
            "cover",
            "best_repeat",
            "top_k",
            "cover_vote_last",
            "max_cover_milp",
            "best_max_cover_milp",
            "best_vote_global",
            "vote_global",
            "vote_global_milp",
        ),
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
        "max_cover_milp": select_max_cover_milp,
        "best_max_cover_milp": lambda pool, count: select_best_max_cover_milp(
            pool, count, root
        ),
        "cover_vote_last": select_cover_vote_last,
        "best_vote_global": lambda pool, count: select_best_vote_global(pool, count, root),
        "vote_global": select_vote_global,
        "vote_global_milp": select_vote_global_milp,
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
