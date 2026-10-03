#!/usr/bin/env python3
"""Aggregate per-skill rollout results with answer voting.

The script reads an existing multi-skill rollout directory.  It does not call
the model or rerun any questions.  Dataset-specific answer extraction,
normalisation, and scoring live behind a small handler registry so that the
voting pipeline can be reused for other datasets.

Example::

    python3 test-phase/test_vote_result.py \
        --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_190420_cover_numskills5 \
        --dataset searchqa
"""
from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class DatasetHandler:
    """Dataset-specific operations used by the generic voting pipeline."""

    extract_answer: Callable[[dict], str]
    normalize_answer: Callable[[str], str]
    evaluate_answer: Callable[[str, list[str]], dict]


@dataclass(frozen=True)
class SkillResults:
    rank: int
    name: str
    path: Path
    rows_by_id: dict[str, dict]


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

    return DatasetHandler(
        extract_answer=extract_answer,
        normalize_answer=normalize_answer,
        evaluate_answer=evaluate,
    )


DATASET_HANDLERS: dict[str, Callable[[], DatasetHandler]] = {
    "searchqa": _searchqa_handler,
}


def get_dataset_handler(dataset: str) -> DatasetHandler:
    """Return the canonical handler used by both voting entry points."""
    try:
        factory = DATASET_HANDLERS[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc
    return factory()


def discover_skill_dirs(input_root: Path) -> list[tuple[int, Path]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist or is not a directory: {input_root}")

    found: list[tuple[int, Path]] = []
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
    skills: list[SkillResults] = []
    expected_ids: set[str] | None = None

    for rank, skill_dir in discover_skill_dirs(input_root):
        rows = load_jsonl(skill_dir / "results.jsonl")
        rows_by_id: dict[str, dict] = {}
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
            missing = sorted(expected_ids - ids)
            extra = sorted(ids - expected_ids)
            raise ValueError(
                f"inconsistent question IDs in {skill_dir.name}: "
                f"missing={len(missing)} extra={len(extra)}"
            )

        skills.append(
            SkillResults(
                rank=rank,
                name=skill_dir.name,
                path=skill_dir,
                rows_by_id=rows_by_id,
            )
        )
        log(f"[load] rank={rank} skill={skill_dir.name} results={len(rows_by_id)}")

    return skills


def is_valid_vote(row: dict, answer: str, handler: DatasetHandler) -> bool:
    if not answer:
        return False
    if row.get("agent_ok") is False:
        return False
    if row.get("phase") in {"timeout", "error"}:
        return False
    return bool(handler.normalize_answer(answer))


def score_mean(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(field, 0.0) or 0.0) for row in rows) / len(rows)


def build_vote_row(
    question_id: str,
    skills: list[SkillResults],
    handler: DatasetHandler,
) -> dict:
    source_rows = [skill.rows_by_id[question_id] for skill in skills]
    first_row = source_rows[0]
    gold_answers = first_row.get("gold_answers")
    if not isinstance(gold_answers, list):
        gold_answers = first_row.get("gold_answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
    gold_answers = [str(answer) for answer in gold_answers]

    answers: list[dict] = []
    vote_counts: Counter[str] = Counter()
    first_answer_for_key: dict[str, str] = {}
    valid_vote_count = 0
    timeout_count = 0

    for skill, row in zip(skills, source_rows):
        answer = handler.extract_answer(row)
        normalized = handler.normalize_answer(answer) if answer else ""
        valid = is_valid_vote(row, answer, handler)
        phase = str(row.get("phase") or "")
        if phase == "timeout":
            timeout_count += 1
        if valid:
            valid_vote_count += 1
            vote_counts[normalized] += 1
            first_answer_for_key.setdefault(normalized, answer)
        answers.append(
            {
                "rank": skill.rank,
                "skill": skill.name,
                "answer": answer,
                "normalized_answer": normalized,
                "valid_vote": valid,
                "phase": phase or None,
            }
        )

    if not vote_counts:
        voted_answer = ""
        status = "no_vote"
        tied_answers: list[str] = []
        vote_count = 0
    else:
        highest = max(vote_counts.values())
        winning_keys = {key for key, count in vote_counts.items() if count == highest}
        tied_answers = [first_answer_for_key[key] for key in vote_counts if key in winning_keys]
        status = "tie" if len(winning_keys) > 1 else "ok"
        voted_answer = next(
            answer["answer"]
            for answer in answers
            if answer["valid_vote"]
            and answer["normalized_answer"] in winning_keys
        )
        vote_count = highest

    evaluation = handler.evaluate_answer(voted_answer, gold_answers)
    return {
        "id": question_id,
        "question": first_row.get("question", ""),
        "gold_answers": gold_answers,
        "answers": answers,
        "vote_counts": dict(vote_counts),
        "voted_answer": voted_answer,
        "vote_count": vote_count,
        "valid_vote_count": valid_vote_count,
        "timeout_count": timeout_count,
        "status": status,
        "tie_answers": tied_answers,
        "tie_break": "first_rank" if status == "tie" else None,
        "em": evaluation["em"],
        "f1": evaluation["f1"],
        "sub_em": evaluation["sub_em"],
        "hard": int(evaluation["em"]),
        "soft": evaluation["f1"],
    }


def default_output_root(input_root: Path) -> Path:
    return input_root.with_name(input_root.name + "_vote")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_HANDLERS))
    parser.add_argument("--output-root")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(input_root)
    )
    if output_root.exists():
        raise FileExistsError(
            f"output root already exists: {output_root}; choose another path or remove it explicitly"
        )

    handler = get_dataset_handler(args.dataset)
    skills = load_skill_results(input_root)
    question_ids = sorted(skills[0].rows_by_id)
    log(f"[vote] dataset={args.dataset} skills={len(skills)} questions={len(question_ids)}")

    vote_rows: list[dict] = []
    covered_ids: set[str] = set()
    tie_count = 0
    no_vote_count = 0
    timeout_count = 0
    for question_id in question_ids:
        row = build_vote_row(question_id, skills, handler)
        vote_rows.append(row)
        if row["status"] == "tie":
            tie_count += 1
        elif row["status"] == "no_vote":
            no_vote_count += 1
        timeout_count += row["timeout_count"]
        if any(
            int(skill.rows_by_id[question_id].get("hard", 0) or 0) == 1
            for skill in skills
        ):
            covered_ids.add(question_id)

    output_root.mkdir(parents=True)
    with (output_root / "vote_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in vote_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_skill = []
    for skill in skills:
        rows = list(skill.rows_by_id.values())
        per_skill.append(
            {
                "rank": skill.rank,
                "skill": skill.name,
                "n_results": len(rows),
                "hard": score_mean(rows, "hard"),
                "soft": score_mean(rows, "soft"),
                "timeout_count": sum(1 for row in rows if row.get("phase") == "timeout"),
                "error_count": sum(1 for row in rows if row.get("phase") == "error"),
            }
        )

    summary = {
        "dataset": args.dataset,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "n_skills": len(skills),
        "n_questions": len(question_ids),
        "per_skill": per_skill,
        "voted": {
            "hard": score_mean(vote_rows, "hard"),
            "soft": score_mean(vote_rows, "soft"),
            "tie_count": tie_count,
            "no_vote_count": no_vote_count,
            "timeout_count": timeout_count,
        },
        "coverage": len(covered_ids) / max(len(question_ids), 1),
        "covered_questions": len(covered_ids),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_root / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "input_root": str(input_root),
                "skills": [
                    {"rank": skill.rank, "name": skill.name, "source": str(skill.path)}
                    for skill in skills
                ],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    log(
        f"[done] voted_hard={summary['voted']['hard']:.4f} "
        f"voted_soft={summary['voted']['soft']:.4f} "
        f"coverage={summary['coverage']:.4f} ties={tie_count} "
        f"no_vote={no_vote_count} saved={output_root}"
    )


if __name__ == "__main__":
    main()
