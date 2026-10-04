"""Shared loading and dataset utilities for SearchQA test scripts."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULT_ROOT = REPO_ROOT / "outputs" / (
    "skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315"
)
VALIDATION_IDS_PATH = REPO_ROOT / "data" / "searchqa_split" / "val" / "items.json"


@dataclass(frozen=True)
class DatasetHandler:
    extract_answer: Callable[[dict], str]
    normalize_answer: Callable[[str], str]
    evaluate_answer: Callable[[str, list[str]], dict]


@dataclass(frozen=True)
class SkillResults:
    rank: int
    name: str
    path: Path
    rows_by_id: dict[str, dict]


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
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _searchqa_handler() -> DatasetHandler:
    from skillopt.envs.searchqa.evaluator import evaluate, normalize_answer

    def extract_answer(row: dict) -> str:
        answer = row.get("predicted_answer", "")
        return answer.strip() if isinstance(answer, str) else ""

    return DatasetHandler(extract_answer, normalize_answer, evaluate)


DATASET_HANDLERS: dict[str, Callable[[], DatasetHandler]] = {
    "searchqa": _searchqa_handler,
}


def get_dataset_handler(dataset: str) -> DatasetHandler:
    try:
        return DATASET_HANDLERS[dataset]()
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc


def is_valid_vote(row: dict, answer: str, handler: DatasetHandler) -> bool:
    if not answer or row.get("agent_ok") is False:
        return False
    if row.get("phase") in {"timeout", "error"}:
        return False
    return bool(handler.normalize_answer(answer))


def build_answer_groups(
    source_rows: list[dict],
    skills: list[SkillResults],
    handler: DatasetHandler,
) -> tuple[list[dict], Counter[str], dict[str, str], tuple[int, ...]]:
    """Extract the shared answer/group representation used by test and gate."""
    answers: list[dict] = []
    vote_counts: Counter[str] = Counter()
    first_answer_for_key: dict[str, str] = {}
    invalid_indices: list[int] = []
    for index, (skill, row) in enumerate(zip(skills, source_rows)):
        answer = handler.extract_answer(row)
        normalized = handler.normalize_answer(answer) if answer else ""
        valid = is_valid_vote(row, answer, handler)
        if valid:
            vote_counts[normalized] += 1
            first_answer_for_key.setdefault(normalized, answer)
        else:
            invalid_indices.append(index)
        answers.append({
            "rank": skill.rank,
            "skill": skill.name,
            "answer": answer,
            "normalized_answer": normalized,
            "valid_vote": valid,
            "phase": str(row.get("phase") or "") or None,
        })
    return answers, vote_counts, first_answer_for_key, tuple(invalid_indices)


def get_gold_answers(row: dict) -> list[str]:
    gold_answers = row.get("gold_answers")
    if not isinstance(gold_answers, list):
        gold_answers = row.get("gold_answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
    return [str(answer) for answer in gold_answers]


def discover_skill_dirs(input_root: Path) -> list[tuple[int, Path]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist or is not a directory: {input_root}")
    found = []
    for path in input_root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"(\d{3})_(.+)", path.name)
        if match and (path / "results.jsonl").is_file():
            found.append((int(match.group(1)), path))
    found.sort(key=lambda pair: pair[0])
    if not found:
        raise RuntimeError(f"no ranked skill result directories found under {input_root}")
    ranks = [rank for rank, _ in found]
    if len(set(ranks)) != len(ranks):
        raise RuntimeError(f"duplicate skill ranks under {input_root}: {ranks}")
    return found


def load_skill_results(input_root: Path) -> list[SkillResults]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist or is not a directory: {input_root}")
    selection_path = input_root / "selection.json"
    if any(path.is_dir() and re.fullmatch(r"\d{3}_.+", path.name) for path in input_root.iterdir()):
        discovered = discover_skill_dirs(input_root)
    else:
        if not selection_path.is_file():
            raise RuntimeError(f"no ranked skill result directories found under {input_root}")
        with selection_path.open(encoding="utf-8") as handle:
            selection = json.load(handle)
        items = selection.get("skills") or selection.get("selected") or []
        if not items:
            raise RuntimeError(f"selection.json has no selected skills: {selection_path}")
        discovered = []
        for item in items:
            source = Path(str(item["source"]))
            if not source.is_dir() or not (source / "results.jsonl").is_file():
                raise FileNotFoundError(f"selected skill results not found: {source}")
            match = re.match(r"(\d{3})_", str(item.get("name", "")))
            rank = int(match.group(1)) if match else len(discovered) + 1
            discovered.append((rank, source))
    skills = []
    expected_ids: set[str] | None = None
    for rank, skill_dir in discovered:
        rows = load_jsonl(skill_dir / "results.jsonl")
        rows_by_id = {}
        for row in rows:
            if row.get("id") is None:
                raise ValueError(f"{skill_dir / 'results.jsonl'} contains a row without id")
            question_id = str(row["id"])
            if question_id in rows_by_id:
                raise ValueError(f"duplicate id={question_id} in {skill_dir / 'results.jsonl'}")
            rows_by_id[question_id] = row
        ids = set(rows_by_id)
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"inconsistent question IDs in {skill_dir.name}")
        skills.append(SkillResults(rank, skill_dir.name, skill_dir, rows_by_id))
        log(f"[load] rank={rank} skill={skill_dir.name} results={len(rows_by_id)}")
    return skills


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
            log(f"[select] skip {name}: incomplete validation results rows={len(rows)} unique_ids={len(result_ids)} expected={len(expected_ids)}")
            continue
        seen_hashes[content_hash] = name
        correct_ids = frozenset(qid for qid, row in records_by_id.items() if int(row.get("hard", 0)) == 1)
        candidate = Candidate(name, skill_path, results_path, records_by_id, correct_ids, version_of(name))
        candidates.append(candidate)
        log(f"[select] loaded {name}: n={len(records_by_id)} hard={len(correct_ids)} acc={candidate.accuracy:.4f}")
    if not candidates:
        raise RuntimeError(f"no complete valid_seen candidates found under {root}")
    return candidates


def complete_records(results_path: Path, expected_ids: frozenset[str]) -> dict[str, dict] | None:
    rows = load_jsonl(results_path)
    records = {str(row["id"]): row for row in rows if row.get("id") is not None}
    if len(records) != len(rows) or frozenset(records) != expected_ids:
        return None
    return records


def load_last_best_skill(root: Path, candidates: list[Candidate]) -> Candidate:
    best_path = root / "best_skill.md"
    if not best_path.is_file():
        raise FileNotFoundError(f"best skill file not found: {best_path}")
    expected_ids = load_expected_validation_ids()
    best_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()
    matches = []
    for source_index, (name, skill_path, results_path) in enumerate(candidate_sources(root)):
        if hashlib.sha256(skill_path.read_bytes()).hexdigest() != best_hash:
            continue
        records = complete_records(results_path, expected_ids)
        if records is not None:
            matches.append((source_index, name, results_path, records))
    if not matches:
        raise RuntimeError(f"no complete valid_seen result matches best skill: {best_path}")
    _, source_name, results_path, records = matches[-1]
    correct_ids = frozenset(qid for qid, row in records.items() if int(row.get("hard", 0) or 0) == 1)
    log(f"[select] best_skill={best_path} matched_runs={len(matches)} using_last={source_name} results={results_path} hard={len(correct_ids)}/{len(records)} acc={len(correct_ids) / max(len(records), 1):.4f}")
    return Candidate("best_skill", best_path, results_path, records, correct_ids, version_of(source_name))
