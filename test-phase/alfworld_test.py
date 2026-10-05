#!/usr/bin/env python3
"""Evaluate ALFWorld skills with best-repeat or top-k selection."""
from __future__ import annotations

import sys
from pathlib import Path

TEST_PHASE_ROOT = Path(__file__).resolve().parent
if str(TEST_PHASE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_PHASE_ROOT))

from utils import run_dataset_test  # noqa: E402


if __name__ == "__main__":
    run_dataset_test(
        dataset="alfworld",
        validation_split="alfworld_path_split",
    )
