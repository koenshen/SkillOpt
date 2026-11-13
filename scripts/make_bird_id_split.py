"""Build the BIRD train/val/test ID split from the dev and mini-dev prompt files.

Split rule (fixed, seed 42):
  * test  = every question in mini-dev (500).
  * val   = 200 questions from dev minus mini-dev, stratified by
            (db_id, difficulty) so its mix matches the remaining dev pool.
  * train = the rest of dev minus mini-dev (834).

Only ``question_id``, ``db_id`` and ``difficulty`` are read, so the inputs can
be the full ``*_prompt.generated.jsonl`` files or a JSONL holding just those
three fields.  The output holds IDs only; ``scripts/materialize_bird.py``
turns it into runnable splits on the machine that has the BIRD data.

Example::

    python scripts/make_bird_id_split.py \
        --dev-jsonl /path/to/datas/dev_prompt.generated.jsonl \
        --mini-dev-jsonl /path/to/datas/mini_dev_prompt.generated.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--mini-dev-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "bird_id_split")
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("question_id", "db_id", "difficulty"):
                if key not in row:
                    raise ValueError(f"{path}:{line_number} is missing {key!r}")
            rows.append({k: row[k] for k in ("question_id", "db_id", "difficulty")})
    ids = [str(r["question_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has duplicate question_id values")
    return rows


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Proportional allocation per (db_id, difficulty), largest remainder."""
    if size > len(rows):
        raise ValueError(f"val size {size} > pool size {len(rows)}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["db_id"], row["difficulty"])].append(row)
    keys = sorted(groups)
    quotas = {k: size * len(groups[k]) / len(rows) for k in keys}
    alloc = {k: int(quotas[k]) for k in keys}
    leftover = size - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), k))[:leftover]:
        alloc[k] += 1
    rng = random.Random(seed)
    picked = []
    for k in keys:
        members = sorted(groups[k], key=lambda r: int(r["question_id"]))
        picked.extend(rng.sample(members, alloc[k]))
    return sorted(picked, key=lambda r: int(r["question_id"]))


def build_split(dev: list[dict], mini: list[dict], val_size: int, seed: int) -> dict[str, list[dict]]:
    dev_ids = {str(r["question_id"]) for r in dev}
    mini_ids = {str(r["question_id"]) for r in mini}
    missing = sorted(mini_ids - dev_ids, key=int)
    if missing:
        raise ValueError(f"mini-dev has {len(missing)} question_id values not in dev, e.g. {missing[:5]}")
    pool = [r for r in dev if str(r["question_id"]) not in mini_ids]
    val = stratified_sample(pool, val_size, seed)
    val_ids = {str(r["question_id"]) for r in val}
    train = [r for r in pool if str(r["question_id"]) not in val_ids]
    test = sorted(mini, key=lambda r: int(r["question_id"]))
    return {"train": train, "val": val, "test": test}


def write_split(split: dict[str, list[dict]], output_dir: Path, args: argparse.Namespace) -> None:
    for name in SPLITS:
        path = output_dir / name / "items.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        items = [{"id": str(r["question_id"])} for r in split[name]]
        path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "benchmark": "BIRD",
        "manifest_type": "id_split",
        "source_files": {"dev": args.dev_jsonl.name, "mini_dev": args.mini_dev_jsonl.name},
        "source_id_field": "question_id",
        "rule": (
            "test = all of mini-dev; val = stratified sample by (db_id, difficulty) "
            "from dev minus mini-dev; train = the rest of dev minus mini-dev"
        ),
        "seed": args.seed,
        "counts": {name: len(split[name]) for name in SPLITS},
        "difficulty": {
            name: dict(sorted(Counter(r["difficulty"] for r in split[name]).items())) for name in SPLITS
        },
        "db_id": {name: dict(sorted(Counter(r["db_id"] for r in split[name]).items())) for name in SPLITS},
        "item_fields": ["id"],
        "notes": [
            "This is an ID manifest, not the BIRD payload.",
            "Run scripts/materialize_bird.py to build data/bird_split from the prompt files.",
            "test rows come from the mini-dev file (its SQL/evidence differ from dev for a few questions).",
        ],
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    split = build_split(load_rows(args.dev_jsonl), load_rows(args.mini_dev_jsonl), args.val_size, args.seed)
    write_split(split, args.output_dir, args)
    print({name: len(split[name]) for name in SPLITS}, "->", args.output_dir)


if __name__ == "__main__":
    main()
