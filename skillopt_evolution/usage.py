"""Durable provider-reported token counters for one sequential evolution run."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from skillopt.model import get_token_summary


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def zero_usage() -> dict:
    return dict(calls=0, input_tokens=0, output_tokens=0, total_tokens=0)


@contextmanager
def measure_usage(directory: Path, stage: str, *, attempt: str | None = None):
    """Measure one optimizer call or one adapter batch; no overlapping contexts."""
    path = (Path(directory) / "token_events" / f"{uuid4().hex}.json").resolve()
    event = {"stage": stage, "status": "in_flight", "started_utc": now()}
    if attempt is not None:
        event["attempt"] = attempt
    save_json(path, event)
    before = get_token_summary()["_total"].copy()
    start = time.monotonic()
    try:
        yield path
    except BaseException as error:
        event.update(status="failed", error_type=type(error).__name__)
        raise
    else:
        event["status"] = "finished"
    finally:
        after = get_token_summary()["_total"]
        event["usage"] = {
            public: after[key] - before[key]
            for public, key in (("calls", "calls"), ("input_tokens", "prompt_tokens"),
                                ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens"))
        }
        event.update(finished_utc=now(), elapsed_seconds=round(time.monotonic() - start, 3))
        save_json(path, event)


def summarize_usage(directory: Path) -> dict:
    result = {"generation": zero_usage(), "validation": zero_usage(), "total": zero_usage(),
              "unfinished_events": [], "failed_events": 0, "zero_reported_token_events": []}
    for path in sorted(directory.rglob("token_events/*.json")):
        event = read_json(path)
        if event["status"] == "in_flight":
            result["unfinished_events"].append(str(path.relative_to(directory)))
            continue
        result["failed_events"] += int(event["status"] == "failed")
        amounts = event["usage"]
        if not amounts["total_tokens"]:
            result["zero_reported_token_events"].append(str(path.relative_to(directory)))
        for key, value in amounts.items():
            result[event["stage"]][key] += value
            result["total"][key] += value
    return result
