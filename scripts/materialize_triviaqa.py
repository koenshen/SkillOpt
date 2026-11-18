"""Materialize TriviaQA (rc.web) into SkillOpt's SearchQA item format.

The ``rc.web`` configuration pairs each question with (a) the matched
Wikipedia entity page and (b) eight web search snippets.  Context is built
from both, in the ``[DOC]`` format the SearchQA adapter truncates on.

Difficulty is controlled by the answer type (``WikipediaEntity`` needs an
exact entity name, ``Numerical``/``FreeForm``/``Range`` do not) and by the
question source.  Both are known to differ a lot between contiguous slices of
the released file, so the split is stratified on ``(answer_type, source)``.

The released TriviaQA test split has placeholder answers (``<unk>``), so this
script splits the validation split instead; the train split is not used.

Usage::

    python scripts/materialize_triviaqa.py \
        --output-dir data/triviaqa_split \
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

DEFAULT_DATASET = "mandarjoshi/trivia_qa"
DEFAULT_CONFIG = "rc.web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="rc.web | rc")
    parser.add_argument("--source-split", default="validation",
                        help="validation (the test split has <unk> answers)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "triviaqa_split")
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _answers(row: dict) -> list[str]:
    answer = row.get("answer") or {}
    candidates = [answer.get("value"), *list(answer.get("aliases") or [])]
    out: list[str] = []
    for cand in candidates:
        text = str(cand or "").strip()
        if text and text != "<unk>" and text not in out:
            out.append(text)
    return out


def _context_text(row: dict) -> str:
    blocks: list[str] = []
    pages = row.get("entity_pages") or {}
    for title, body in zip(
        list(pages.get("title") or []), list(pages.get("wiki_context") or [])
    ):
        blocks.append(doc_block(title, body))
    results = row.get("search_results") or {}
    for title, body in zip(
        list(results.get("title") or []), list(results.get("search_context") or [])
    ):
        blocks.append(doc_block(title, body))
    return join_doc_blocks(blocks)


def normalize_row(row: dict) -> dict:
    answers = _answers(row)
    if not answers:
        raise ValueError(f"TriviaQA row {row.get('question_id')!r} has no usable answer")
    answer_type = str((row.get("answer") or {}).get("type") or "unknown")
    return {
        "id": str(row["question_id"]),
        "question": str(row["question"]).strip(),
        "context": _context_text(row),
        "answers": answers,
        # kept for inspection only; the adapter reads just the four fields above
        "task_type": answer_type,
        "source": str(row.get("question_source") or "").rstrip("/").split("/")[-1] or "unknown",
    }


def difficulty_key(item: dict) -> str:
    return f"{item['task_type']}/{item['source']}"


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
    print(f"  usable items: {len(items)} (skipped {skipped}: placeholder answer or empty context)")

    splits = stratified_split(items, sizes, difficulty_key, seed=args.seed)
    check_splits(splits, difficulty_key, tol=0.02)
    counts = write_splits(
        args.output_dir,
        splits,
        source=f"{args.dataset} [{args.config_name}] split={args.source_split}",
        benchmark="triviaqa",
        extra_manifest={
            "config_name": args.config_name,
            "source_split": args.source_split,
            "seed": args.seed,
            "train_size": args.train,
            "val_size": args.val,
            "test_size": args.test,
            "difficulty_key": "answer_type/source",
        },
    )
    print(f"Wrote TriviaQA splits to {args.output_dir.resolve()}: {counts}")
    for name, share in sorted(stratum_report(splits, difficulty_key).items()):
        print(f"  {name}: " + ", ".join(f"{k}={v:.2%}" for k, v in share.items()))


if __name__ == "__main__":
    main()
