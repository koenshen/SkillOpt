"""Materialize the released LiveMathematicianBench ID split."""

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
from skillopt.envs.livemathematicianbench.dataloader import load_items  # noqa: E402

RELEASED_MONTHS = ("202511", "202512", "202601", "202602")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "livemathematicianbench_id_split",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "livemathematicianbench" / "data",
        help="Directory containing monthly qa_*_final.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "livemathematicianbench_split",
    )
    return parser.parse_args()


def materialize(args: argparse.Namespace) -> dict[str, int]:
    split_ids = load_manifest_ids(args.manifest_dir)
    _reject_duplicate_manifest_ids(split_ids)
    wanted = {item_id for ids in split_ids.values() for item_id in ids}

    by_id: dict[str, dict] = {}
    for month in RELEASED_MONTHS:
        source = args.source_dir / month / f"qa_{month}_final.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing released LiveMath source file: {source}")
        for row in load_items(str(source)):
            key = str(row["id"])
            if key in by_id:
                raise ValueError(f"Duplicate LiveMath source id: {key}")
            by_id[key] = row

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
        "source_months": list(RELEASED_MONTHS),
        "source_dir": str(args.source_dir.resolve()),
        "counts": counts,
        "item_fields": sorted(by_id[next(iter(wanted))].keys()),
    }
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return counts


def main() -> None:
    args = parse_args()
    counts = materialize(args)
    print(f"Wrote LiveMathematicianBench splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
