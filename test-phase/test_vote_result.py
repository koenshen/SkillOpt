#!/usr/bin/env python3
"""Aggregate per-skill rollout results with answer voting.

The majority and coalition policies read existing rollout results only.  The
rag policy additionally performs the explicitly requested retrieval and
re-inference on disagreement questions.  Dataset-specific answer extraction,
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
import math
import os
import re
import sys
import time
import urllib.request
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


def _evaluate_answer(handler: DatasetHandler, answer: str, row: dict) -> dict:
    if handler.evaluate_row is not None:
        return handler.evaluate_row(answer, row)
    return handler.evaluate_answer(answer, get_gold_answers(row))


def _candidate_groups(row: dict) -> list[tuple[str, tuple[int, ...]]]:
    groups = []
    for key in row["vote_counts"]:
        members = tuple(
            sorted(
                index
                for index, answer in enumerate(row["answers"])
                if answer["valid_vote"] and answer["normalized_answer"] == key
            )
        )
        groups.append((key, members))
    return sorted(groups, key=lambda item: (-len(item[1]), item[1]))


def _choose_scored_answer(row: dict, scores: dict[str, float | None]) -> tuple[str, dict]:
    """Choose a scored candidate while preserving the existing majority fallback."""
    candidates = [
        (key, score)
        for key, _ in _candidate_groups(row)
        if (score := scores.get(key)) is not None
    ]
    if not candidates:
        return row["voted_answer"], {"score": None, "support": 0, "fallback": "majority"}
    best_score = max(score for _, score in candidates)
    best = [key for key, score in candidates if score == best_score]
    if len(best) != 1:
        return row["voted_answer"], {"score": best_score, "support": 0, "fallback": "majority", "score_tie": True}
    key = best[0]
    answer = next(
        (item["answer"] for item in row["answers"] if item["normalized_answer"] == key),
        row["voted_answer"],
    )
    return answer, {"score": best_score, "support": 0, "fallback": None, "score_tie": False}


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

    evaluation = _evaluate_answer(handler, voted_answer, first_row)
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
    dataset: str = "searchqa",
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

    expected_ids = None
    if dataset != "searchqa":
        from scripts.eval_only import get_adapter
        from utils import load_split_ids

        with (gate_root / "config.json").open(encoding="utf-8") as handle:
            gate_cfg = json.load(handle)
        if gate_cfg.get("env") != dataset:
            raise ValueError(f"gate config does not identify dataset {dataset}")
        adapter = get_adapter(gate_cfg)
        adapter.setup(gate_cfg)
        expected_ids = load_split_ids(adapter, "valid_seen", gate_cfg.get("seed", 42))
    candidates = load_candidates(gate_root, expected_ids)
    by_name = {candidate.name: candidate for candidate in candidates}
    gate_candidates = []
    for name in selected_names:
        base_name = re.sub(r"^\d{3}_", "", name)
        if base_name == "best_skill":
            gate_candidates.append(load_last_best_skill(gate_root, candidates, expected_ids))
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
            correct = int(_evaluate_answer(handler, answer, source_rows[0])["em"])
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


def _gate_correctness(
    gate_skills: list[SkillResults], handler: DatasetHandler
) -> tuple[list[str], dict[str, list[int]]]:
    question_ids = sorted(gate_skills[0].rows_by_id)
    correctness = {skill.name: [] for skill in gate_skills}
    for question_id in question_ids:
        gold = get_gold_answers(gate_skills[0].rows_by_id[question_id])
        for skill in gate_skills:
            row = skill.rows_by_id[question_id]
            answer = handler.extract_answer(row)
            valid = is_valid_vote(row, answer, handler)
            correctness[skill.name].append(
                int(valid and _evaluate_answer(handler, answer, row)["em"])
            )
    return question_ids, correctness


def _required_env(names: tuple[str, ...]) -> dict[str, str]:
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("missing required environment variable(s): " + ", ".join(missing))
    return values


def _retry_request(operation, label: str, stats: dict):
    """Initial request plus five retries; the caller decides the fallback."""
    for attempt in range(6):
        stats["attempts"] += 1
        if attempt:
            stats["retries"] += 1
            log(f"[{label}] retry={attempt}/5 starting")
        try:
            return operation()
        except Exception as exc:
            # Avoid printing HTTP headers/bodies, which may contain credentials.
            log(f"[{label}] attempt={attempt + 1}/6 failed ({type(exc).__name__})")
            if attempt < 5:
                time.sleep(min(2 * (attempt + 1), 10))
    stats["exhausted"] += 1
    log(f"[{label}] five retries exhausted")
    return None


def _embedding(text: str, env: dict[str, str], label: str, stats: dict):
    def request_embedding():
        request = urllib.request.Request(
            env["EMBEDDING_BASE_URL"],
            data=json.dumps({"model": env["EMBEDDING_MODEL"], "input": text}).encode("utf-8"),
            headers={"Authorization": f"Bearer {env['EMBEDDING_API_KEY']}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        data = body["data"]
        if len(data) != 1:
            raise ValueError("expected one embedding")
        vector = [float(value) for value in data[0]["embedding"]]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("invalid embedding")
        norm = math.sqrt(sum(value * value for value in vector))
        if not norm:
            raise ValueError("zero embedding")
        return [value / norm for value in vector]
    return _retry_request(request_embedding, f"embedding {label}", stats)


def _top_gate_indices(query: list[float], vectors: list[list[float]], k: int) -> list[int]:
    if any(len(vector) != len(query) for vector in vectors):
        raise ValueError("embedding dimensions differ")
    # Vectors are normalised: their dot product is cosine similarity.
    return sorted(range(len(vectors)),
                  key=lambda i: (-sum(a * b for a, b in zip(query, vectors[i])), i))[:k]


def _reliability_answers(row: dict, reliabilities: list[float]) -> dict:
    groups = _candidate_groups(row)
    output = {}
    for mode in ("max", "mean"):
        scores = {}
        for key, members in groups:
            values = [reliabilities[index] for index in members]
            scores[key] = max(values) if mode == "max" else sum(values) / len(values)
        answer, detail = _choose_scored_answer(row, scores)
        output[mode] = (answer, {**detail, "candidate_scores": scores})
    return output


def _source_selection(input_root: Path) -> tuple[Path, dict]:
    """Follow aggregated outputs back to the existing rollout selection."""
    seen = set()
    while input_root not in seen:
        seen.add(input_root)
        with (input_root / "selection.json").open(encoding="utf-8") as handle:
            selection = json.load(handle)
        if selection.get("selected"):
            return input_root, selection
        source = selection.get("input_root")
        if not source:
            break
        input_root = Path(source).expanduser().resolve()
    raise ValueError("cannot locate original rollout selection.json")


def _with_method_answer(row: dict, answer: str, method: str,
                        handler: DatasetHandler, detail: dict) -> dict:
    result = dict(row)
    result["base_voted_answer"] = row["voted_answer"]
    result["base_hard"] = row["hard"]
    result["voted_answer"] = answer
    result["method"] = method
    result["decision"] = detail
    evaluation = _evaluate_answer(handler, answer, row)
    result.update(em=evaluation["em"], f1=evaluation["f1"],
                  sub_em=evaluation["sub_em"], hard=int(evaluation["em"]),
                  soft=evaluation["f1"])
    return result


def _rerun_one_skill(adapter, item: dict, skill: SkillResults, content: str,
                     examples: str | None, handler: DatasetHandler, out_dir: Path,
                     log_prefix: str = "rag") -> dict:
    stats = {"attempts": 0, "retries": 0, "exhausted": 0,
             "timeout_count": 0, "error_count": 0}
    def run():
        attempt_dir = out_dir / f"attempt_{stats['attempts']:02d}"
        try:
            if examples:
                # Reuse SearchQA's existing diagnostic prompt path so the
                # examples are appended while the selected skill stays unchanged.
                results = adapter.rollout(
                    [item], content, str(attempt_dir), diagnostic_mode=True,
                    diagnostic_instruction=examples,
                )
            else:
                # No-RAG control: use the ordinary rollout prompt unchanged.
                results = adapter.rollout([item], content, str(attempt_dir))
        except Exception as exc:
            timeout = "timeout" in str(exc).lower() or isinstance(exc, TimeoutError)
            stats["timeout_count" if timeout else "error_count"] += 1
            raise
        if len(results) != 1:
            stats["error_count"] += 1
            raise RuntimeError("expected one rollout result")
        row = results[0]
        if not is_valid_vote(row, handler.extract_answer(row), handler):
            timeout = row.get("phase") == "timeout" or "timeout" in str(row.get("fail_reason", "")).lower()
            stats["timeout_count" if timeout else "error_count"] += 1
            raise RuntimeError("invalid rollout answer")
        return row
    result = _retry_request(run, f"{log_prefix} id={item['id']} skill={skill.rank}", stats)
    if result is None:
        log(f"[{log_prefix}] id={item['id']} skill={skill.rank}: excluded as invalid vote")
        result = {"id": str(item["id"]), "question": item["question"],
                  "gold_answers": item.get("answers", [item.get("ground_truth", "")]), "predicted_answer": "",
                  "phase": "error", "agent_ok": False}
    return {**result, "request_stats": stats}


def _is_unanimous_vote(row: dict, skill_count: int) -> bool:
    return (
        row.get("status") not in {"tie", "no_vote"}
        and row.get("valid_vote_count") == skill_count
        and row.get("vote_count") == skill_count
    )


def _run_rag_bundle(input_root: Path, gate_skills: list[SkillResults],
                    skills: list[SkillResults], base_rows: list[dict],
                    handler: DatasetHandler, cfg: dict,
                    output_root: Path, use_rag: bool = True,
                    matrix_mode: bool = False, matrix_samples: int = 1,
                    retrieval_k: int | None = None,
                    primary_method: str = "rag") -> tuple[dict, dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import hashlib
    from scripts.eval_only import get_adapter
    from utils import configure_runtime

    embedding_env = None
    if use_rag:
        embedding_env = _required_env(("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"))
    configure_runtime(cfg)
    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    items_by_id = {
        str(item["id"]): item
        for item in adapter.build_eval_env(0, "valid_unseen", cfg.get("seed", 42))
    }
    gate_items_by_id = {
        str(item["id"]): item
        for item in adapter.build_eval_env(0, "valid_seen", cfg.get("seed", 42))
    }

    _, selection = _source_selection(input_root)
    selected = selection["selected"]
    if len(selected) != len(skills):
        raise ValueError("selected skill count differs from rollout count")
    contents = []
    for item, gate_skill in zip(selected, gate_skills):
        path = Path(item["skill_path"])
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(gate_skill.path.read_bytes()).digest():
            raise ValueError(f"gate skill content mismatch: {gate_skill.name}")
        contents.append(path.read_text(encoding="utf-8"))

    k = len(skills)
    if retrieval_k is None:
        retrieval_k = k
    if retrieval_k <= 0:
        raise ValueError("retrieval_k must be positive")
    missing = set(row["id"] for row in base_rows) - items_by_id.keys()
    if missing:
        raise ValueError(f"test IDs missing from dataset: {sorted(missing)}")

    ordered_gate_ids = []
    matrix = []
    global_reliability = []
    if use_rag:
        gate_ids, correctness = _gate_correctness(gate_skills, handler)
        if len(gate_ids) < k:
            raise ValueError(f"gate questions={len(gate_ids)} smaller than K={k}")
        ordered_gate_ids = list(gate_skills[0].rows_by_id)
        correctness_index = {qid: index for index, qid in enumerate(gate_ids)}
        matrix = [
            [correctness[skill.name][correctness_index[qid]] for qid in ordered_gate_ids]
            for skill in gate_skills
        ]
        global_reliability = [sum(values) / len(ordered_gate_ids) for values in matrix]

    output_root.mkdir(parents=True, exist_ok=False)
    embedding_stats = {"attempts": 0, "retries": 0, "exhausted": 0}
    disagreement = [row for row in base_rows if len(row["vote_counts"]) > 1]
    gate_vectors: list[list[float]] | None = [] if use_rag else None
    if use_rag and disagreement:
        for index, qid in enumerate(ordered_gate_ids, 1):
            vector = _embedding(
                gate_skills[0].rows_by_id[qid]["question"],
                embedding_env,
                f"gate {index}/{len(ordered_gate_ids)}",
                embedding_stats,
            )
            if vector is None:
                gate_vectors = None
                log("[rag] gate embedding failed; QA-RAG and local methods fallback to majority")
                break
            gate_vectors.append(vector)
            log(f"[embedding] gate {index}/{len(ordered_gate_ids)} completed")

    if matrix_mode and not use_rag:
        raise ValueError("matrix_mode requires use_rag=True")
    output = (
        {name: [] for name in (primary_method, "global_max", "global_mean", "local_max", "local_mean")}
        if use_rag and not matrix_mode else
        {"rag_matrix": []} if matrix_mode else {"no_rag": []}
    )
    contexts: dict[str, dict] = {}
    mode_label = primary_method if use_rag else "no_rag"
    for position, row in enumerate(disagreement, 1):
        log(f"[{mode_label}] preparing disagreement {position}/{len(disagreement)} id={row['id']}")
        decisions = {}
        if use_rag and not matrix_mode:
            global_decisions = _reliability_answers(row, global_reliability)
            decisions = {
                f"global_{mode}": global_decisions[mode]
                for mode in ("max", "mean")
            }
        context = {
            "row": row,
            "decisions": decisions,
            "run_rag": not use_rag,
            "item": items_by_id[row["id"]],
        }
        if use_rag and gate_vectors is not None:
            query = _embedding(
                row["question"], embedding_env, f"test id={row['id']}", embedding_stats
            )
            if query is not None:
                indices = _top_gate_indices(query, gate_vectors, retrieval_k)
                neighbours = [ordered_gate_ids[index] for index in indices]
                if not matrix_mode:
                    local_reliability = [
                        sum(values[index] for index in indices) / retrieval_k for values in matrix
                    ]
                    local_decisions = _reliability_answers(row, local_reliability)
                    for mode in ("max", "mean"):
                        answer, detail = local_decisions[mode]
                        decisions[f"local_{mode}"] = (
                            answer,
                            {**detail, "gate_ids": neighbours,
                             "skill_reliabilities": local_reliability},
                        )
                examples = "\n\n".join(
                    handler.format_example(
                        gate_items_by_id[qid], gate_skills[0].rows_by_id[qid]
                    )
                    if handler.format_example is not None else
                    f"Example question:\n{gate_skills[0].rows_by_id[qid]['question']}\n\n"
                    f"Example answer:\n{get_gold_answers(gate_skills[0].rows_by_id[qid])[0]}"
                    for qid in neighbours
                )
                context.update(
                    item=items_by_id[row["id"]],
                    examples=examples,
                    neighbours=neighbours,
                    run_rag=True,
                )
            else:
                log(f"[rag] id={row['id']}: embedding unavailable; local and QA-RAG fallback to majority")
        contexts[row["id"]] = context

    # Submit every (question, skill) pair together. This removes the previous
    # per-question barrier while leaving each retry and rollout implementation
    # unchanged.
    reruns_by_question: dict[str, dict[int, dict]] = {}
    future_map = {}
    with ThreadPoolExecutor(max_workers=max(1, int(adapter.workers))) as executor:
        for question_id, context in contexts.items():
            if not context["run_rag"]:
                continue
            for skill, content in zip(skills, contents):
                future = executor.submit(
                    _rerun_one_skill,
                    adapter,
                    context["item"],
                    skill,
                    content,
                    context.get("examples", ""),
                    handler,
                    output_root / "reinference" / f"{skill.rank:03d}_{skill.name}" / question_id,
                    mode_label,
                )
                future_map[future] = (question_id, skill.rank)
        for completed, future in enumerate(as_completed(future_map), 1):
            question_id, rank = future_map[future]
            reruns_by_question.setdefault(question_id, {})[rank] = future.result()
            if completed % max(1, len(skills)) == 0 or completed == len(future_map):
                log(f"[rag] reinference tasks {completed}/{len(future_map)} completed")

    third_reruns_by_question: dict[str, dict[tuple[int, int], dict]] = {}
    third_questions: set[str] = set()
    stage2_unanimous_questions: set[str] = set()
    if matrix_mode:
        # First inspect the second-round RAG vote. Only non-unanimous cases
        # receive the additional K * matrix_samples rollouts.
        for question_id, context in contexts.items():
            if not context["run_rag"]:
                continue
            reruns = [
                reruns_by_question[question_id][skill.rank]
                for skill in skills
            ]
            temporary = [
                SkillResults(skill.rank, skill.name, skill.path, {question_id: result})
                for skill, result in zip(skills, reruns)
            ]
            second_vote = build_vote_row(question_id, temporary, handler)
            context["second_vote"] = second_vote
            if _is_unanimous_vote(second_vote, len(skills)):
                stage2_unanimous_questions.add(question_id)
            else:
                third_questions.add(question_id)

        third_future_map = {}
        with ThreadPoolExecutor(max_workers=max(1, int(adapter.workers))) as executor:
            for question_id in third_questions:
                context = contexts[question_id]
                for skill, content in zip(skills, contents):
                    for sample_index in range(matrix_samples):
                        future = executor.submit(
                            _rerun_one_skill,
                            adapter,
                            context["item"],
                            skill,
                            content,
                            context["examples"],
                            handler,
                            output_root / "reinference" / "stage3"
                            / f"{skill.rank:03d}_{skill.name}"
                            / question_id / f"sample_{sample_index + 1:02d}",
                            "rag_matrix",
                        )
                        third_future_map[future] = (question_id, skill.rank, sample_index)
            for completed, future in enumerate(as_completed(third_future_map), 1):
                question_id, rank, sample_index = third_future_map[future]
                third_reruns_by_question.setdefault(question_id, {})[(rank, sample_index)] = future.result()
                if completed % max(1, len(skills)) == 0 or completed == len(third_future_map):
                    log(
                        f"[rag_matrix] stage3 tasks "
                        f"{completed}/{len(third_future_map)} completed"
                    )
    else:
        third_future_map = {}

    for row in base_rows:
        decisions = {
            name: (row["voted_answer"], {"fallback": None, "reason": "unchanged"})
            for name in output
        }
        if len(row["vote_counts"]) > 1:
            context = contexts[row["id"]]
            decisions.update(context["decisions"])
            if context["run_rag"]:
                reruns = [
                    reruns_by_question[row["id"]][skill.rank]
                    for skill in skills
                ]
                temporary = [
                    SkillResults(skill.rank, skill.name, skill.path, {row["id"]: result})
                    for skill, result in zip(skills, reruns)
                ]
                new_vote = build_vote_row(row["id"], temporary, handler)
                exhausted = new_vote["status"] == "no_vote"
                if exhausted:
                    log(f"[{ 'rag' if use_rag else 'no_rag' }] id={row['id']}: all reinferences failed; fallback to original majority")
                if matrix_mode:
                    second_vote = context["second_vote"]
                    if _is_unanimous_vote(second_vote, len(skills)):
                        final_answer = second_vote["voted_answer"]
                        detail = {
                            "stage": 2,
                            "stop_reason": "rag_unanimous",
                            "fallback": None,
                            "gate_ids": context["neighbours"],
                            "second_vote": second_vote,
                            "rollout_results": reruns,
                        }
                    else:
                        third_reruns = [
                            third_reruns_by_question[row["id"]][(skill.rank, sample_index)]
                            for skill in skills
                            for sample_index in range(matrix_samples)
                        ]
                        third_temporary = [
                            SkillResults(
                                skill.rank * 100000 + sample_index,
                                f"{skill.name}__sample_{sample_index + 1:02d}",
                                skill.path,
                                {
                                    row["id"]: third_reruns_by_question[row["id"]][
                                        (skill.rank, sample_index)
                                    ]
                                },
                            )
                            for skill in skills
                            for sample_index in range(matrix_samples)
                        ]
                        third_vote = build_vote_row(row["id"], third_temporary, handler)
                        third_exhausted = third_vote["status"] == "no_vote"
                        final_answer = (
                            second_vote["voted_answer"]
                            if third_exhausted else third_vote["voted_answer"]
                        )
                        if third_exhausted:
                            log(
                                f"[rag_matrix] id={row['id']}: all stage3 reinferences "
                                "failed; fallback to stage2 RAG vote"
                            )
                        detail = {
                            "stage": 2 if third_exhausted else 3,
                            "stop_reason": "stage3_no_vote" if third_exhausted else "stage3_vote",
                            "fallback": "stage2_rag" if third_exhausted else None,
                            "gate_ids": context["neighbours"],
                            "second_vote": second_vote,
                            "third_vote": third_vote,
                            "rollout_results": reruns + third_reruns,
                        }
                    decisions["rag_matrix"] = (final_answer, detail)
                else:
                    method_name = primary_method if use_rag else "no_rag"
                    decisions[method_name] = (
                        row["voted_answer"] if exhausted else new_vote["voted_answer"],
                        {"fallback": "majority" if exhausted else None,
                         **({"gate_ids": context["neighbours"]} if use_rag else {}),
                         "new_vote": new_vote,
                         "rollout_results": reruns},
                    )
            else:
                method_name = "rag_matrix" if matrix_mode else primary_method
                decisions[method_name] = (
                    row["voted_answer"],
                    {"fallback": "majority", "reason": "embedding_failure"},
                )
        for name, (answer, detail) in decisions.items():
            # The matrix policy keeps only its final result file; the shared
            # context may still contain auxiliary RAG decisions.
            if name not in output:
                continue
            output[name].append(_with_method_answer(row, answer, name, handler, detail))

    return output, {
        "embedding": embedding_stats,
        "L": retrieval_k,
        "matrix_mode": matrix_mode,
        "matrix_samples": matrix_samples if matrix_mode else None,
        "disagreement_count": len(disagreement),
        "stage2_unanimous_count": len(stage2_unanimous_questions),
        "stage3_question_count": len(third_questions),
        "stage3_tasks": len(third_future_map),
        "global_skill_reliabilities": global_reliability,
        "embedding_model": embedding_env["EMBEDDING_MODEL"] if embedding_env else None,
        "inference_model": os.environ["OPENAI_COMPATIBLE_MODEL"],
        "reinference_tasks": len(future_map),
        "max_workers": max(1, int(adapter.workers)),
    }

def _method_summary(rows: list[dict]) -> dict:
    disagreement = [row for row in rows if len(row["vote_counts"]) > 1]
    unanimous = [row for row in rows if row["vote_count"] == len(row["answers"])]
    reruns = [result for row in rows for result in row["decision"].get("rollout_results", [])]
    return {
        "hard": score_mean(rows, "hard"), "soft": score_mean(rows, "soft"),
        "disagreement_count": len(disagreement),
        "disagreement_hard": score_mean(disagreement, "hard"),
        "disagreement_soft": score_mean(disagreement, "soft"),
        "unanimous_count": len(unanimous),
        "unanimous_hard": score_mean(unanimous, "hard"),
        "unanimous_soft": score_mean(unanimous, "soft"),
        "changed_from_majority": sum(row["voted_answer"] != row["base_voted_answer"] for row in rows),
        "unchanged": sum(row["voted_answer"] == row["base_voted_answer"] for row in rows),
        "rescue": sum(not row["base_hard"] and row["hard"] for row in rows),
        "harm": sum(row["base_hard"] and not row["hard"] for row in rows),
        "fallback_count": sum(row["decision"].get("fallback") == "majority" for row in rows),
        "score_tie_count": sum(bool(row["decision"].get("score_tie")) for row in rows),
        "reinference_count": len(reruns),
        "retry_count": sum(result["request_stats"]["retries"] for result in reruns),
        "timeout_count": sum(result["request_stats"]["timeout_count"] for result in reruns),
        "error_count": sum(result["request_stats"]["error_count"] for result in reruns),
        "exhausted_count": sum(result["request_stats"]["exhausted"] for result in reruns),
    }


def default_output_root(input_root: Path, policy: str) -> Path:
    suffix = {
        "coalition": "_coalition",
        "rag": "_rag",
        "rag_plus": "_rag_plus",
        "no_rag": "_no_rag",
        "rag_matrix": "_rag_matrix",
    }.get(policy, "_vote")
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
    parser.add_argument(
        "--policy",
        choices=("majority", "coalition", "rag", "rag_plus", "no_rag", "rag_matrix"),
        default="majority",
    )
    parser.add_argument("--matrix-samples", type=int)
    parser.add_argument("--override-margin", type=float)
    args = parser.parse_args()

    if args.policy == "rag_matrix":
        if args.matrix_samples is None:
            parser.error("--matrix-samples is required with --policy rag_matrix")
        if args.matrix_samples <= 0:
            parser.error("--matrix-samples must be positive")
    elif args.matrix_samples is not None:
        parser.error("--matrix-samples is only valid with --policy rag_matrix")

    if args.policy in {"coalition", "rag", "rag_plus", "no_rag", "rag_matrix"}:
        if not args.gate_root:
            if args.policy in {"rag", "rag_plus", "no_rag", "rag_matrix"}:
                _, selection = _source_selection(Path(args.input_root).expanduser().resolve())
                args.gate_root = selection.get("source_result_root")
            if not args.gate_root:
                parser.error("--gate-root is required when the source selection does not identify gate data")
    if args.policy == "coalition":
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
    if args.policy in {"coalition", "rag", "rag_plus", "no_rag", "rag_matrix"}:
        gate_root = Path(args.gate_root).expanduser().resolve()
        gate_skills = _selected_gate_skills(input_root, gate_root, skills, args.dataset)
        if args.policy == "coalition":
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

    rag_outputs = None
    rag_metadata = None
    if args.policy in {"rag", "rag_plus", "no_rag", "rag_matrix"}:
        gate_config_path = Path(args.gate_root).expanduser().resolve() / "config.json"
        if not gate_config_path.is_file():
            raise FileNotFoundError(f"missing gate config.json: {gate_config_path}")
        with gate_config_path.open(encoding="utf-8") as handle:
            gate_cfg = json.load(handle)
        rag_outputs, rag_metadata = _run_rag_bundle(
            input_root, gate_skills, skills, vote_rows, handler, gate_cfg, output_root,
            use_rag=args.policy in {"rag", "rag_plus", "rag_matrix"},
            matrix_mode=args.policy == "rag_matrix",
            matrix_samples=args.matrix_samples or 1,
            retrieval_k=7 if args.policy == "rag_plus" else len(skills),
            primary_method=args.policy,
        )

    if args.policy not in {"rag", "rag_plus", "no_rag", "rag_matrix"}:
        output_root.mkdir(parents=True)
    with (output_root / "vote_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in vote_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rag_outputs is not None:
        for method, rows in rag_outputs.items():
            with (output_root / f"{method}_results.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
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
    if args.policy in {"rag", "rag_plus", "no_rag", "rag_matrix"}:
        summary[args.policy] = {method: _method_summary(rows) for method, rows in rag_outputs.items()}
        summary[f"{args.policy}_metadata"] = {
            **rag_metadata, "gate_root": str(Path(args.gate_root).resolve())
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
