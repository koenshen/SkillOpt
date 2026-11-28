"""Evaluate the five nominated single edits on the archived 200-item TriviaQA dev split."""

import argparse
import fcntl
import json
from pathlib import Path

from .evaluate import evaluate_candidate
from .run import archived_adapter, check_or_write, fingerprint, sha
from .usage import read_json, save_json, summarize_usage


PARENT_SHA = "7ca91e72edc0d0874fce618fa43f20a9e47c61214b63f47a6806dd39cfe037ab"
EDITS = ((74, 3), (53, 2), (35, 1), (22, 2), (12, 3))


def transplant(source, parent, step, index):
    folder = source / "steps" / f"step_{step:04}"
    edit = read_json(folder / "ranked_edits.json")["edits"][index - 1]
    applied = read_json(folder / "edit_apply_report.json")
    if not any(row["index"] == index and row["status"].startswith("applied") for row in applied):
        raise ValueError(f"Unapplied source edit at step {step}, index {index}")
    anchor, full = edit["target"].strip(), edit["content"].strip()
    if edit["op"] != "insert_after" or parent.splitlines().count(anchor) != 1:
        raise ValueError(f"Expected a unique whole-line anchor for {step}/{index}")
    if anchor + "\n\n" + full not in (folder / "candidate_skill.md").read_text():
        raise ValueError(f"Cannot verify insertion in the archived child for {step}/{index}")
    addition = full
    if step == 35:
        addition = full.splitlines()[0]
    if step == 22:
        marker = "; for broad-category questions "
        if full.count(marker) != 1:
            raise ValueError("Cannot locate the nominated v22 clause")
        addition = "- for broad-category questions " + full.split(marker)[1]
    if not addition or addition in parent:
        raise ValueError(f"Missing or redundant addition for {step}/{index}")
    lines = parent.splitlines(keepends=True)
    position = next(i for i, line in enumerate(lines) if line.rstrip("\r\n") == anchor)
    lines.insert(position + 1, "\n" + addition + "\n")
    return "".join(lines), {"step": step, "index": index, "anchor": anchor, "addition": addition}


def evaluate(source, out):
    cfg = read_json(source / "config.json")
    split = source.parent.parent / cfg["split_dir"] / "val" / "items.json"
    items = read_json(split)
    parent = (source / "best_skill.md").read_text()
    parent_dir = source / "steps/step_0062"
    if sha(parent) != PARENT_SHA or parent != (parent_dir / "candidate_skill.md").read_text():
        raise ValueError("Source does not contain the expected v62")
    sheet = [json.loads(line) for line in (parent_dir / "selection_eval/results.jsonl").read_text().splitlines()]
    rows = {row["id"]: row for row in sheet}
    ids = [item["id"] for item in items]
    if (len(ids) != 200 or len(set(ids)) != 200 or len(sheet) != 200 or set(ids) != set(rows)
            or not all(row["agent_ok"] for row in sheet) or sum(row["hard"] for row in sheet) != 178):
        raise ValueError("Expected the fixed dev split and original 178/200 v62 sheet")
    children = {}
    proposals = {}
    for step, index in EDITS:
        name = f"v{step}_edit{index}"
        child, spec = transplant(source, parent, step, index)
        children[name] = child
        proposals[name] = dict(spec, sha256=sha(child))
    if len(set(children.values())) != len(EDITS):
        raise ValueError("Duplicate child text")
    check_or_write(out / "manifest.json", dict(source=str(source), parent_sha256=PARENT_SHA,
                   dev_file=str(split), dev_sha256=fingerprint(items), config_sha256=fingerprint(cfg),
                   model=cfg["target_model"], workers=cfg["workers"], max_tokens=cfg["max_completion_tokens"],
                   proposals=proposals))
    adapter = archived_adapter(cfg)
    scores = {}
    for name, child in children.items():
        directory = out / name
        directory.mkdir(exist_ok=True)
        (directory / "skill.md").write_text(child)
        result = evaluate_candidate(adapter=adapter, dev_items=items, order=ids, skill=child,
                                    parent_rows=rows, cached_rows={}, out_dir=directory, batch_size=200,
                                    require_complete=True)
        scores[name] = {key: result[key] for key in ("correct", "n_observed", "fixes", "regressions")}
        print(name, result["correct"], "/ 200", flush=True)
    save_json(out / "summary.json", dict(scores=scores, usage=summarize_usage(out)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New output directory, or the same run to resume")
    args = parser.parse_args()
    args.out.mkdir(mode=0o700, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(args.source.resolve(), args.out.resolve())
