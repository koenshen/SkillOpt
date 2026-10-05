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
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PHASE_ROOT = Path(__file__).resolve().parent
if str(TEST_PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_PHASE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import (  # noqa: E402
    DATASET_HANDLERS,
    build_answer_groups,
    DatasetHandler,
    SkillResults,
    get_dataset_handler,
    get_gold_answers,
    is_valid_vote,
    load_jsonl,
    load_skill_results,
    log,
)


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
    gold_answers = get_gold_answers(first_row)

    answers, vote_counts, first_answer_for_key, invalid_indices = build_answer_groups(
        source_rows, skills, handler
    )
    valid_vote_count = len(answers) - len(invalid_indices)
    timeout_count = sum(
        1 for row in source_rows if str(row.get("phase") or "") == "timeout"
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
        "invalid_indices": list(invalid_indices),
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

    # Reuse the shared candidate discovery and best-skill matching utilities.
    from utils import load_candidates, load_last_best_skill

    candidates = load_candidates(gate_root)
    by_name = {candidate.name: candidate for candidate in candidates}
    gate_candidates = []
    for name in selected_names:
        base_name = re.sub(r"^\d{3}_", "", name)
        if base_name == "best_skill":
            gate_candidates.append(load_last_best_skill(gate_root, candidates))
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
        answers, vote_counts, _, invalid = build_answer_groups(
            source_rows, gate_skills, handler
        )
        groups = [
            tuple(
                index
                for index, answer in enumerate(answers)
                if answer["valid_vote"] and answer["normalized_answer"] == key
            )
            for key in vote_counts
        ]
        partition = _partition_key(groups, invalid)
        for group in groups:
            answer = answers[group[0]]["answer"]
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
    majority_key = next(
        (
            key
            for key, count in row["vote_counts"].items()
            if count == majority_vote
        ),
        "",
    )
    scored = [candidate for candidate in candidates if candidate["score"] is not None]
    if not scored:
        return {
            "majority_answer": row["voted_answer"],
            "majority_override_answer": row["voted_answer"],
            "pure_reliability_answer": row["voted_answer"],
            "coalition_candidates": candidates,
            "overridden": False,
            "coalition_score_tie": False,
            "coalition_fallback": "majority",
        }

    best_score = max(candidate["score"] for candidate in scored)
    best = [candidate for candidate in scored if candidate["score"] == best_score]
    majority_candidate = next((candidate for candidate in candidates if candidate["key"] == majority_key), None)
    score_tie = len(best) > 1
    if score_tie:
        # Reuse build_vote_row's existing majority and first-rank tie policy.
        pure_answer = row["voted_answer"]
        coalition_fallback = "majority"
    else:
        pure_key = best[0]["key"]
        pure_answer = next(
            (
                item["answer"]
                for item in row["answers"]
                if item["normalized_answer"] == pure_key
            ),
            row["voted_answer"],
        )
        coalition_fallback = None
    override_answer = row["voted_answer"]
    if not score_tie and majority_candidate and majority_candidate["score"] is not None:
        challenger = best[0]
        if (
            challenger["key"] != majority_key
            and challenger["score"] - majority_candidate["score"] > override_margin
        ):
            override_answer = pure_answer
    return {
        "majority_answer": row["voted_answer"],
        "majority_override_answer": override_answer,
        "pure_reliability_answer": pure_answer,
        "coalition_candidates": candidates,
        "overridden": override_answer != row["voted_answer"],
        "coalition_score_tie": score_tie,
        "coalition_fallback": coalition_fallback,
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
