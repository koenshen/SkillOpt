"""Materialize the released OfficeQA UID split with full question records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_searchqa import (  # noqa: E402
    SPLITS,
    _reject_duplicate_manifest_ids,
    load_manifest_ids,
)


def _split_field(value: str) -> list[str]:
    """Normalize the newline-separated document fields in the CSV."""
    return [part.strip() for part in str(value or "").replace("\r", "").split("\n") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "officeqa_id_split",
        help="Directory containing the released train/val/test UID manifests.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "officeqa" / "officeqa_full.csv",
        help="The downloaded OfficeQA full CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "officeqa_split",
        help="Output directory consumed by configs/officeqa/default.yaml.",
    )
    return parser.parse_args()


def materialize(args: argparse.Namespace) -> dict[str, int]:
    split_ids = load_manifest_ids(args.manifest_dir)
    _reject_duplicate_manifest_ids(split_ids)
    wanted = {item_id for ids in split_ids.values() for item_id in ids}

    by_id: dict[str, dict] = {}
    with args.source_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"uid", "question", "answer", "source_docs", "source_files", "difficulty"}
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"OfficeQA CSV is missing columns: {', '.join(missing_columns)}")
        for row in reader:
            uid = str(row.get("uid") or "").strip()
            if not uid:
                raise ValueError("OfficeQA source row is missing uid")
            if uid in by_id:
                raise ValueError(f"Duplicate OfficeQA source uid: {uid}")
            answer = str(row.get("answer") or "").strip()
            by_id[uid] = {
                "id": uid,
                "uid": uid,
                "question": str(row.get("question") or "").strip(),
                "ground_truth": answer,
                "answers": [answer] if answer else [],
                "task_type": str(row.get("difficulty") or "officeqa").strip() or "officeqa",
                "category": str(row.get("difficulty") or "officeqa").strip() or "officeqa",
                "source_files": _split_field(row.get("source_files", "")),
                "source_docs": _split_field(row.get("source_docs", "")),
            }

    missing = sorted(wanted - by_id.keys())
    if missing:
        raise RuntimeError(f"Missing {len(missing)} manifest IDs: {', '.join(missing[:5])}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        output_split = args.output_dir / split
        output_split.mkdir(parents=True, exist_ok=True)
        items = []
        for uid in split_ids[split]:
            item = dict(by_id[uid])
            item["split"] = split
            items.append(item)
        with (output_split / "items.json").open("w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        counts[split] = len(items)

    manifest = {
        "benchmark": "OfficeQA",
        "manifest_type": "materialized_id_split",
        "source_manifest_dir": str(args.manifest_dir.resolve()),
        "source_dataset": str(args.source_path.resolve()),
        "counts": counts,
        "item_fields": sorted(next(iter(by_id.values())).keys()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return counts


def main() -> None:
    args = parse_args()
    counts = materialize(args)
    print(f"Wrote OfficeQA splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
