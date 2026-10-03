"""Materialize the released SpreadsheetBench ID split."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "spreadsheetbench_id_split",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "spreadsheetbench_verified_400" / "dataset.json",
        help="Extracted SpreadsheetBench dataset.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "spreadsheetbench_split",
    )
    return parser.parse_args()


def materialize(args: argparse.Namespace) -> dict[str, int]:
    split_ids = load_manifest_ids(args.manifest_dir)
    _reject_duplicate_manifest_ids(split_ids)
    with args.source_path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {args.source_path}")

    by_id: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError("SpreadsheetBench source row is missing id")
        key = str(row["id"])
        if key in by_id:
            raise ValueError(f"Duplicate SpreadsheetBench source id: {key}")
        by_id[key] = row

    wanted = {item_id for ids in split_ids.values() for item_id in ids}
    missing = sorted(wanted - by_id.keys())
    if missing:
        raise RuntimeError(f"Missing {len(missing)} manifest IDs: {', '.join(missing[:5])}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        output_split = args.output_dir / split
        output_split.mkdir(parents=True, exist_ok=True)
        items = [by_id[item_id] for item_id in split_ids[split]]
        with (output_split / "items.json").open("w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        counts[split] = len(items)

    manifest = {
        "source_manifest_dir": str(args.manifest_dir.resolve()),
        "source_dataset": str(args.source_path.resolve()),
        "counts": counts,
        "item_fields": sorted(by_id[next(iter(wanted))].keys()),
    }
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return counts


def main() -> None:
    args = parse_args()
    counts = materialize(args)
    print(f"Wrote SpreadsheetBench splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
