"""Materialize HotpotQA (distractor) into SkillOpt's SearchQA item format.

HotpotQA's distractor setting ships ten passages per question: two that
support the answer and eight distractors drawn from the same paragraphs.  No
external retrieval is needed, which matches SkillOpt's single-turn open-book
SearchQA adapter.

Difficulty is controlled by task type (``bridge`` needs two hops,
``comparison`` needs two entities), so the split is stratified on
``(type, level)`` to keep all three splits at the same mix.

Usage::

    python scripts/materialize_hotpotqa.py \
        --output-dir data/hotpotqa_split \
        --train 1600 --val 200 --test 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qa_split_utils import (  # noqa: E402
    check_splits,
    doc_block,
    join_doc_blocks,
    stratified_split,
    stratum_report,
    write_splits,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG = "distractor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="distractor | fullwiki")
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "hotpotqa_split")
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _context_text(context: dict) -> str:
    titles = list(context.get("title") or [])
    sentence_lists = list(context.get("sentences") or [])
    blocks = [
        doc_block(title, " ".join(str(s) for s in (sentences or [])))
        for title, sentences in zip(titles, sentence_lists)
    ]
    return join_doc_blocks(blocks)


def normalize_row(row: dict) -> dict:
    answers = [str(row.get("answer") or "").strip()]
    answers = [a for a in answers if a]
    if not answers:
        raise ValueError(f"HotpotQA row {row.get('id')!r} has no answer")
    return {
        "id": str(row["id"]),
        "question": str(row["question"]).strip(),
        "context": _context_text(row.get("context") or {}),
        "answers": answers,
        # kept for inspection only; the adapter reads just the four fields above
        "task_type": str(row.get("type") or "hotpotqa"),
        "level": str(row.get("level") or "hard"),
    }


def difficulty_key(item: dict) -> str:
    return f"{item['task_type']}/{item['level']}"


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Missing dependency 'datasets'. Install with:\n"
            "  python -m pip install 'skillopt[searchqa]'"
        ) from exc

    sizes = {"train": args.train, "val": args.val, "test": args.test}
    total = sum(sizes.values())
    print(f"Loading {args.dataset} [{args.config_name}] split={args.source_split} ...")
    rows = load_dataset(args.dataset, args.config_name, split=args.source_split)
    print(f"  pool size: {len(rows)} (need {total})")
    if len(rows) < total:
        raise SystemExit(f"pool has {len(rows)} rows, need {total}")

    items = []
    skipped = 0
    for row in rows:
        try:
            item = normalize_row(row)
        except ValueError:
            skipped += 1
            continue
        if not item["context"]:
            skipped += 1
            continue
        items.append(item)
    print(f"  usable items: {len(items)} (skipped {skipped}: empty answer or context)")

    splits = stratified_split(items, sizes, difficulty_key, seed=args.seed)
    check_splits(splits, difficulty_key, tol=0.02)
    counts = write_splits(
        args.output_dir,
        splits,
        source=f"{args.dataset} [{args.config_name}] split={args.source_split}",
        benchmark="hotpotqa",
        extra_manifest={
            "config_name": args.config_name,
            "source_split": args.source_split,
            "seed": args.seed,
            "train_size": args.train,
            "val_size": args.val,
            "test_size": args.test,
            "difficulty_key": "task_type/level",
        },
    )
    print(f"Wrote HotpotQA splits to {args.output_dir.resolve()}: {counts}")
    for name, share in sorted(stratum_report(splits, difficulty_key).items()):
        print(f"  {name}: " + ", ".join(f"{k}={v:.2%}" for k, v in share.items()))


if __name__ == "__main__":
    main()
