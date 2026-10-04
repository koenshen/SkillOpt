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
import itertools
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
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist or is not a directory: {input_root}")
    skills: list[SkillResults] = []
    expected_ids: set[str] | None = None

    # A previous vote run stores the selected skill source directories in
    # selection.json.  Reuse those paths when input_root is the *_vote folder.
    selection_path = input_root / "selection.json"
    if not any(path.is_dir() and re.fullmatch(r"\d{3}_.+", path.name) for path in input_root.iterdir()):
        if not selection_path.is_file():
            raise RuntimeError(f"no ranked skill result directories found under {input_root}")
        with selection_path.open(encoding="utf-8") as handle:
            selection = json.load(handle)
        selected_items = selection.get("skills") or selection.get("selected") or []
        if not selected_items:
            raise RuntimeError(f"selection.json has no selected skills: {selection_path}")
        discovered = []
        for item in selected_items:
            source = Path(str(item["source"]))
            if not source.is_dir() or not (source / "results.jsonl").is_file():
                raise FileNotFoundError(f"selected skill results not found: {source}")
            rank_match = re.match(r"(\d{3})_", str(item.get("name", "")))
            rank = int(rank_match.group(1)) if rank_match else len(discovered) + 1
            discovered.append((rank, source))
    else:
        discovered = discover_skill_dirs(input_root)

    for rank, skill_dir in discovered:
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


def get_gold_answers(row: dict) -> list[str]:
    """Read reference answers, including timeout rows' legacy field."""
    gold_answers = row.get("gold_answers")
    if not isinstance(gold_answers, list):
        gold_answers = row.get("gold_answer", [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
    return [str(answer) for answer in gold_answers]


def build_vote_row(
    question_id: str,
    skills: list[SkillResults],
    handler: DatasetHandler,
) -> dict:
    source_rows = [skill.rows_by_id[question_id] for skill in skills]
    first_row = source_rows[0]
    gold_answers = get_gold_answers(first_row)

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


def _red(message: str) -> str:
    return f"\033[31m{message}\033[0m"


def _partition_key(groups: list[tuple[int, ...]], invalid: tuple[int, ...]) -> tuple:
    return (tuple(sorted(groups)), tuple(sorted(invalid)))


def _selected_gate_skills(
    input_root: Path,
    gate_root: Path,
    test_skills: list[SkillResults],
) -> list[SkillResults]:
    """Load only the skills selected in the test output from SkillOpt gate data."""
    selection_path = input_root / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing selection.json: {selection_path}")
    with selection_path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    selected_items = selection.get("skills") or selection.get("selected") or []
    selected_names = [str(item["name"]) for item in selected_items]
    if len(selected_names) != len(test_skills):
        raise ValueError(
            f"selection.json has {len(selected_names)} skills but test has {len(test_skills)}"
        )

    # Reuse SearchQA's existing candidate discovery and best-skill matching.
    try:
        from searchqa_test import _load_last_best_skill, load_candidates
    except ImportError as exc:
        raise RuntimeError("coalition policy requires test-phase/searchqa_test.py") from exc

    candidates = load_candidates(gate_root)
    by_name = {candidate.name: candidate for candidate in candidates}
    gate_candidates = []
    for name in selected_names:
        base_name = re.sub(r"^\d{3}_", "", name)
        if base_name == "best_skill":
            gate_candidates.append(_load_last_best_skill(gate_root, candidates))
            continue
        candidate = by_name.get(base_name)
        if candidate is None:
            raise ValueError(
                f"selected skill {name} was not found in gate root {gate_root}"
            )
        gate_candidates.append(candidate)

    result: list[SkillResults] = []
    for rank, (test_skill, candidate) in enumerate(zip(test_skills, gate_candidates), 1):
        result.append(
            SkillResults(
                rank=rank,
                name=test_skill.name,
                path=candidate.skill_path,
                rows_by_id=candidate.records_by_id,
            )
        )
    return result


def _build_gate_stats(
    gate_skills: list[SkillResults],
    handler: DatasetHandler,
) -> tuple[dict, dict]:
    """Build exact-configuration and coalition reliability counts."""
    exact: dict[tuple, list[int]] = {}
    coalition: dict[tuple[int, ...], list[int]] = {}
    question_ids = sorted(gate_skills[0].rows_by_id)
    for question_id in question_ids:
        source_rows = [skill.rows_by_id[question_id] for skill in gate_skills]
        gold = get_gold_answers(source_rows[0])
        groups_by_answer: dict[str, list[int]] = {}
        invalid: list[int] = []
        answers: list[str] = []
        valid: list[bool] = []
        for index, row in enumerate(source_rows):
            answer = handler.extract_answer(row)
            is_valid = is_valid_vote(row, answer, handler)
            normalized = handler.normalize_answer(answer) if is_valid else ""
            answers.append(answer)
            valid.append(is_valid)
            if is_valid:
                groups_by_answer.setdefault(normalized, []).append(index)
            else:
                invalid.append(index)
        groups = [tuple(sorted(indices)) for indices in groups_by_answer.values()]
        partition = _partition_key(groups, tuple(invalid))
        for group in groups:
            answer = answers[group[0]]
            correct = int(handler.evaluate_answer(answer, gold)["em"])
            entry = exact.setdefault((partition, group), [0, 0])
            entry[0] += 1
            entry[1] += correct
            for size in range(1, len(group) + 1):
                for subset in itertools.combinations(group, size):
                    entry = coalition.setdefault(tuple(subset), [0, 0])
                    entry[0] += 1
                    entry[1] += correct
    return exact, coalition


def _reliability_for_group(
    group: tuple[int, ...],
    partition: tuple,
    exact: dict,
    coalition: dict,
) -> tuple[float | None, int, str]:
    exact_entry = exact.get((partition, group))
    if exact_entry and exact_entry[0] > 0:
        return exact_entry[1] / exact_entry[0], exact_entry[0], "exact"
    full_entry = coalition.get(group)
    if full_entry and full_entry[0] > 0:
        return full_entry[1] / full_entry[0], full_entry[0], "coalition"
    for size in range(len(group) - 1, 0, -1):
        entries = []
        for subset in itertools.combinations(group, size):
            entry = coalition.get(tuple(subset))
            if entry and entry[0] > 0:
                entries.append(entry)
        if entries:
            total = sum(entry[0] for entry in entries)
            correct = sum(entry[1] for entry in entries)
            return correct / total, total, f"subset_{size}"
    return None, 0, "none"


def _choose_coalition_answers(
    row: dict,
    exact: dict,
    coalition: dict,
    override_margin: float,
) -> dict:
    groups_by_key = {
        key: tuple(
            sorted(
                index
                for index, answer in enumerate(row["answers"])
                if answer["valid_vote"] and answer["normalized_answer"] == key
            )
        )
        for key in row["vote_counts"]
    }
    groups = sorted(groups_by_key.items(), key=lambda item: (-len(item[1]), item[1]))
    invalid = tuple(index for index, answer in enumerate(row["answers"]) if not answer["valid_vote"])
    partition = _partition_key([group for _, group in groups], invalid)
    candidates = []
    for key, group in groups:
        score, support, level = _reliability_for_group(group, partition, exact, coalition)
        candidates.append({"key": key, "group": group, "score": score, "support": support, "level": level})

    majority_vote = max(row["vote_counts"].values()) if row["vote_counts"] else 0
    majority_keys = [key for key, count in row["vote_counts"].items() if count == majority_vote]
    majority_key = majority_keys[0] if majority_keys else ""
    scored = [candidate for candidate in candidates if candidate["score"] is not None]
    if not scored:
        return {
            "majority_answer": row["voted_answer"],
            "majority_override_answer": row["voted_answer"],
            "pure_reliability_answer": row["voted_answer"],
            "coalition_candidates": candidates,
            "overridden": False,
        }

    best_score = max(candidate["score"] for candidate in scored)
    best = [candidate for candidate in scored if candidate["score"] == best_score]
    majority_candidate = next((candidate for candidate in candidates if candidate["key"] == majority_key), None)
    if len(best) > 1:
        if len(majority_keys) > 1:
            message = _red(
                f"[coalition tie] id={row['id']} candidates="
                f"{[candidate['key'] for candidate in best]} scores={best_score:.4f}"
            )
            print(message, file=sys.stderr, flush=True)
            raise RuntimeError(f"unresolved coalition tie for question {row['id']}")
        if majority_candidate in best:
            pure_key = majority_key
        else:
            pure_key = majority_key
    else:
        pure_key = best[0]["key"]
    pure_answer = next((item["answer"] for item in row["answers"] if item["normalized_answer"] == pure_key), row["voted_answer"])
    override_answer = row["voted_answer"]
    if majority_candidate and majority_candidate["score"] is not None:
        challenger = next((candidate for candidate in scored if candidate["key"] == pure_key), None)
        if challenger and pure_key != majority_key and challenger["score"] - majority_candidate["score"] > override_margin:
            override_answer = pure_answer
    return {
        "majority_answer": row["voted_answer"],
        "majority_override_answer": override_answer,
        "pure_reliability_answer": pure_answer,
        "coalition_candidates": candidates,
        "overridden": override_answer != row["voted_answer"],
    }


def default_output_root(input_root: Path, policy: str) -> Path:
    suffix = "_coalition" if policy == "coalition" else "_vote"
    base_name = input_root.name
    if base_name.endswith("_vote"):
        base_name = base_name[: -len("_vote")]
    return input_root.with_name(base_name + suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_HANDLERS))
    parser.add_argument("--output-root")
    parser.add_argument("--gate-root")
    parser.add_argument("--policy", choices=("majority", "coalition"), default="majority")
    parser.add_argument("--override-margin", type=float)
    args = parser.parse_args()

    if args.policy == "coalition":
        if not args.gate_root:
            parser.error("--gate-root is required with --policy coalition")
        if args.override_margin is None:
            parser.error("--override-margin is required with --policy coalition")
        if args.override_margin < 0:
            parser.error("--override-margin must be non-negative")

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(input_root, args.policy)
    )
    if output_root.exists():
        raise FileExistsError(
            f"output root already exists: {output_root}; choose another path or remove it explicitly"
        )

    handler = get_dataset_handler(args.dataset)
    skills = load_skill_results(input_root)
    gate_skills = None
    gate_stats = None
    if args.policy == "coalition":
        gate_root = Path(args.gate_root).expanduser().resolve()
        gate_skills = _selected_gate_skills(input_root, gate_root, skills)
        gate_stats = _build_gate_stats(gate_skills, handler)
    question_ids = sorted(skills[0].rows_by_id)
    log(f"[vote] dataset={args.dataset} skills={len(skills)} questions={len(question_ids)}")

    vote_rows: list[dict] = []
    covered_ids: set[str] = set()
    tie_count = 0
    no_vote_count = 0
    timeout_count = 0
    for question_id in question_ids:
        row = build_vote_row(question_id, skills, handler)
        if args.policy == "coalition":
            analysis = _choose_coalition_answers(
                row,
                gate_stats[0],
                gate_stats[1],
                args.override_margin,
            )
            row["coalition"] = analysis
            for field in ("majority_override_answer", "pure_reliability_answer"):
                evaluation = handler.evaluate_answer(analysis[field], row["gold_answers"])
                row[f"{field}_hard"] = int(evaluation["em"])
                row[f"{field}_soft"] = evaluation["f1"]
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
    if args.policy == "coalition":
        summary["coalition"] = {
            "override_margin": args.override_margin,
            "majority_override_hard": score_mean(vote_rows, "majority_override_answer_hard"),
            "majority_override_soft": score_mean(vote_rows, "majority_override_answer_soft"),
            "pure_reliability_hard": score_mean(vote_rows, "pure_reliability_answer_hard"),
            "pure_reliability_soft": score_mean(vote_rows, "pure_reliability_answer_soft"),
            "override_count": sum(1 for row in vote_rows if row["coalition"]["overridden"]),
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

    if args.policy == "coalition":
        with (output_root / "coalition_gate_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "gate_root": str(Path(args.gate_root).expanduser().resolve()),
                    "skills": [skill.name for skill in gate_skills],
                    "exact_configuration_count": len(gate_stats[0]),
                    "coalition_count": len(gate_stats[1]),
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
