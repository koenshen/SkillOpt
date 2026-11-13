"""Materialize runnable BIRD splits from the ID manifest and the prompt files.

train/val rows are read from the dev prompt file and test rows from the
mini-dev prompt file (mini-dev revises SQL/evidence for a few questions).
The source files are only read; output goes to ``--output-dir``.

Example (on the machine with the BIRD data)::

    python scripts/materialize_bird.py \
        --dev-jsonl /path/to/datas/dev_prompt.generated.jsonl \
        --mini-dev-jsonl /path/to/datas/mini_dev_prompt.generated.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = ("question_id", "db_id", "question", "evidence", "SQL", "difficulty", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--mini-dev-jsonl", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data" / "bird_id_split")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "bird_split")
    return parser.parse_args()


def load_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {missing}")
            if not str(row["prompt"]).strip() or not str(row["SQL"]).strip():
                raise ValueError(f"{path}:{line_number} has an empty prompt or SQL")
            key = str(row["question_id"])
            if key in rows:
                raise ValueError(f"{path} has duplicate question_id {key}")
            rows[key] = row
    return rows


def normalize(row: dict) -> dict:
    return {
        "id": str(row["question_id"]),
        "question_id": row["question_id"],
        "db_id": row["db_id"],
        "difficulty": row["difficulty"],
        "task_type": row["difficulty"],
        "question": row["question"],
        "evidence": row["evidence"],
        "prompt": row["prompt"],
        "SQL": row["SQL"],
    }


def materialize(manifest_dir: Path, output_dir: Path, dev: dict[str, dict], mini: dict[str, dict]) -> dict[str, int]:
    sources = {"train": dev, "val": dev, "test": mini}
    seen: dict[str, str] = {}
    counts = {}
    for split in SPLITS:
        ids = [str(x["id"]) for x in json.loads((manifest_dir / split / "items.json").read_text(encoding="utf-8"))]
        for item_id in ids:
            if item_id in seen:
                raise ValueError(f"id {item_id} is in both {seen[item_id]} and {split}")
            seen[item_id] = split
        absent = [i for i in ids if i not in sources[split]]
        if absent:
            raise RuntimeError(f"{split}: {len(absent)} ids not found in the source file, e.g. {absent[:5]}")
        items = [normalize(sources[split][i]) for i in ids]
        out = output_dir / split / "items.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts[split] = len(items)
    return counts


def main() -> None:
    args = parse_args()
    dev = load_by_id(args.dev_jsonl)
    mini = load_by_id(args.mini_dev_jsonl)
    counts = materialize(args.manifest_dir.resolve(), args.output_dir.resolve(), dev, mini)
    manifest = {
        "source_manifest_dir": str(args.manifest_dir.resolve()),
        "source_files": {"train/val": str(args.dev_jsonl.resolve()), "test": str(args.mini_dev_jsonl.resolve())},
        "counts": counts,
        "item_fields": ["id", "question_id", "db_id", "difficulty", "task_type", "question", "evidence", "prompt", "SQL"],
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(counts, "->", args.output_dir)


if __name__ == "__main__":
    main()
