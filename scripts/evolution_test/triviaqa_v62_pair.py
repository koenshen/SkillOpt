"""Fresh paired TriviaQA test for v62 and the v22 single edit frozen in the dev run."""

import argparse
import fcntl
import json
import re
from pathlib import Path

from scripts.evolution_test.scoring import run_test
from skillopt_evolution.run import archived_adapter, check_or_write, fingerprint, sha
from skillopt_evolution.triviaqa_v62_edits import PARENT_SHA
from skillopt_evolution.usage import read_json, save_json


# Ordered first-match rules from the 178/200 vs 178/200 dev analysis.
PATTERNS = (
    ("quantity/date", r"\bhow (?:many|much|old|long|far|often)\b|^when\b|\b(?:what|which) (?:year|day|date|month|decade|century)\b"),
    ("location", r"^where\b|\b(?:which|what) (?:[a-z]+ ){0,2}(?:city|country|town|village|capital|county|state|island|continent|ocean|sea|river)\b"),
    ("kind/type", r"\b(?:what|which) (?:type|kind|sort|class|species|breed|category|family)\b|\b(?:types?|kinds?|sorts?|species|varieties|breeds) of (?:which|what)\b"),
    ("who/whose", r"^who(?:m|se)?\b"),
)


def group(question):
    q = question.lower().strip(" \"'")
    return next((name for name, rule in PATTERNS if re.search(rule, q)), "other")


def sheet(path, items):
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    rows = {r["id"]: r for r in records}
    if len(records) != len(items) or len(rows) != len(items):
        raise ValueError(f"Incomplete or duplicate IDs in {path}")
    for item in items:
        row = rows[item["id"]]
        if not row["agent_ok"] or row["question"] != item["question"] or set(row["gold_answers"]) != set(item["answers"]):
            raise ValueError(f"Invalid or mismatched observation in {path}: {item['id']}")
    return rows


def compare(rows, ids):
    parent, child = rows["v62"], rows["v22_edit2"]
    return {
        "n": len(ids),
        "v62": sum(parent[k]["hard"] for k in ids),
        "v22_edit2": sum(child[k]["hard"] for k in ids),
        "only_v62": [k for k in ids if parent[k]["hard"] > child[k]["hard"]],
        "only_v22": [k for k in ids if child[k]["hard"] > parent[k]["hard"]],
    }


def evaluate(dev, out):
    meta = read_json(dev / "manifest.json")
    source = Path(meta["source"])
    cfg = read_json(source / "config.json")
    child_sha = meta["proposals"]["v22_edit2"]["sha256"]
    paths = {"v62": source / "best_skill.md", "v22_edit2": dev / "v22_edit2/skill.md"}
    skills = {name: path.read_text() for name, path in paths.items()}
    digests = {name: sha(skill) for name, skill in skills.items()}
    if fingerprint(cfg) != meta["config_sha256"] or digests != {"v62": PARENT_SHA, "v22_edit2": child_sha}:
        raise ValueError("Archived model configuration or frozen skill text has changed")
    split = Path(meta["dev_file"]).parent.parent / "test/items.json"
    items = read_json(split)
    ids = [q["id"] for q in items]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Expected 1,000 distinct TriviaQA test IDs")
    v0 = sheet(source / "test_eval_baseline/results.jsonl", items)
    groups = {q["id"]: group(q["question"]) for q in items}
    check_or_write(out / "manifest.json", dict(dev_run=str(dev), split=str(split), test_sha256=fingerprint(items),
                   skills=digests, config_sha256=fingerprint(cfg), model=cfg["target_model"], workers=cfg["workers"],
                   max_tokens=cfg["max_completion_tokens"], groups=groups, v0_correct=[k for k in ids if v0[k]["hard"]]))
    reports, rows = {}, {}
    adapter = archived_adapter(cfg)
    for name, skill in skills.items():
        folder = out / name
        reports[name] = run_test(adapter=adapter, items=items, skill=skill, out_dir=folder)
        save_json(folder / "summary.json", reports[name])
        if reports[name]["n_failed"]:
            raise RuntimeError(f"Unresolved test items for {name}; resume this output to retry")
        rows[name] = sheet(folder / "results.jsonl", items)
    by_group = {name: compare(rows, [k for k in ids if groups[k] == name])
                for name in [key for key, _ in PATTERNS] + ["other"]}
    by_v0 = {str(h): compare(rows, [k for k in ids if v0[k]["hard"] == h]) for h in (0, 1)}
    save_json(out / "summary.json", dict(overall=compare(rows, ids), groups=by_group, by_v0=by_v0, runs=reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Fresh run directory; reuse only to resume it")
    args = parser.parse_args()
    args.out.mkdir(mode=0o700, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(args.dev_run.resolve(), args.out.resolve())
