"""Resolve the exact served prompt and visible answer, including cached observations."""
from __future__ import annotations

import json
from pathlib import Path


class TraceReader:
    def __init__(self, source_root, evolution_root):
        self.source = Path(source_root).resolve() if source_root is not None else None
        self.out = Path(evolution_root).resolve()
        self.records = {}

    def register(self, record):
        self.records[record["id"]] = record

    def __call__(self, record, key):
        key = str(key)
        if Path(key).name != key or key in ("", ".", ".."):
            raise ValueError(f"Invalid question ID {key!r}")
        seen = set()
        while True:
            uid = record["id"]
            if uid in seen:
                raise ValueError(f"Circular score provenance for {uid}/{key}")
            seen.add(uid)
            row = record["rows"].get(key)
            if not row or not row["agent_ok"]:
                raise ValueError(f"No valid observation for {uid}/{key}")
            if "sheet" in record:
                if self.source is None:
                    raise ValueError("Missing first-stage archive root")
                sheet = (self.source / record["sheet"]).resolve()
                if not sheet.is_relative_to(self.source):
                    raise ValueError(f"Score sheet outside archive: {sheet}")
                path = sheet.parent / "predictions" / key
                break
            origin = record["row_sources"][key]
            if origin in self.records:
                previous = self.records[origin]
                if previous["sha256"] != record["sha256"] or row != previous["rows"].get(key):
                    raise ValueError(f"Cached row for {uid}/{key} does not match {origin}")
                record = previous
                continue
            work = (Path(record["source"]) / origin).resolve()
            if not work.is_relative_to(self.out) or work.parent.name != "evaluation_attempts":
                raise ValueError(f"Unknown answer provenance for {uid}/{key}: {origin}")
            path = work / "rollout" / "predictions" / key
            break
        prompt = (path / "target_user_prompt.txt").read_text(encoding="utf-8")
        turns = json.loads((path / "conversation.json").read_text(encoding="utf-8"))
        visible = [t["content"] for t in turns if t.get("type") == "message"]
        if not prompt.strip() or not visible or not all(isinstance(t, str) for t in visible):
            raise ValueError(f"Missing visible input/output at {path}")
        if "response" in row and row["response"] != visible[-1]:
            raise ValueError(f"Score sheet and final visible answer disagree at {path}")
        return {"user_prompt": prompt, "response": "\n\n".join(visible)}
