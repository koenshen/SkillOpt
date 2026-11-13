"""BIRD text-to-SQL dataloader.

Each split directory (train/, val/, test/) holds ``items.json``: a JSON array
written by ``scripts/materialize_bird.py``.  Every item has ``id``,
``db_id``, ``difficulty``, ``prompt`` (schema + evidence + question) and the
gold ``SQL``.
"""
from __future__ import annotations

import json

from skillopt.datasets.base import SplitDataLoader

REQUIRED_FIELDS = ("id", "db_id", "prompt", "SQL")


class BirdDataLoader(SplitDataLoader):
    """Split-dir loader that checks the fields the rollout needs."""

    def load_split_items(self, split_path: str) -> list[dict]:
        items = super().load_split_items(split_path)
        for index, item in enumerate(items):
            missing = [k for k in REQUIRED_FIELDS if not str(item.get(k, "")).strip()]
            if missing:
                raise ValueError(f"{split_path} item {index} is missing {missing}")
            item["id"] = str(item["id"])
        return items

    def load_raw_items(self, data_path: str) -> list[dict]:
        with open(data_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
