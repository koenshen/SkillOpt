"""Render the approved DE-v1 template and read its final, complete child guide."""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import re

from skillopt.optimizer.skill import APPENDIX_END, APPENDIX_START, SLOW_UPDATE_END, SLOW_UPDATE_START

TEMPLATE = Path(__file__).with_name("prompts") / "stage2_de_v1_en.txt"
ROLES = {"A": "Prompt 1 (first historical guide)", "B": "Prompt 2 (second historical guide)",
         "X": "Basic Prompt (guide being improved)", "best": "Prompt 3 (current highest-scoring guide)"}


def template_digest():
    return hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


def render(records, examples):
    evidence = []
    for ex in examples:
        label = "Question to fix" if ex["kind"] == "target" else "Question the Basic Prompt answered correctly"
        evidence.append(f"{label} [{ex['id']}]: {ex['question']}")
        for key, original in ex["served_inputs"].items():
            roles = [ROLES[r] for r, a in ex["answers"].items() if a["input_sha256"] == key]
            evidence.append("Original input for " + ", ".join(roles) + ":\n" + original)
        for r in ("A", "B", "X", "best"):
            if r in ex["answers"]:
                a = ex["answers"][r]
                evidence.append(f"{ROLES[r]} — actual reply ({a['verdict']}, score {a['hard']}):\n{a['visible_response']}")
        evidence.append("")
    subs = {"prompt0": records["X"]["skill"], "prompt1": records["A"]["skill"],
            "prompt2": records["B"]["skill"], "prompt3": records["best"]["skill"],
            "answer_records": "\n".join(evidence)}
    # Single pass: placeholder-like text in real documents is literal data.
    return re.sub(r"<(prompt[0-3]|answer_records)>", lambda m: subs[m[1]], TEMPLATE.read_text(encoding="utf-8"))


def _unchanged_protected_blocks(parent, child):
    for left, right in [(SLOW_UPDATE_START, SLOW_UPDATE_END), (APPENDIX_START, APPENDIX_END)]:
        pattern = re.escape(left) + r".*?" + re.escape(right)
        if (parent.count(left) != child.count(left) or parent.count(right) != child.count(right)
                or re.findall(pattern, parent, re.S) != re.findall(pattern, child, re.S)):
            return False
    return True


def parse_reply(raw, parent, examples):
    """Use the last tagged decision; intermediate evolution steps are audit text."""
    tags = list(re.finditer(r"<(prompt|skip)>((?:(?!</?(?:prompt|skip)>).)*)</\1>", raw, re.S))
    if not tags or raw[tags[-1].end():].strip() not in ("", "```"):
        raise ValueError("Missing final complete <prompt> or <skip> block")
    tag = tags[-1]
    content = tag[2].strip()
    if not content:
        raise ValueError("Empty tagged output")
    if tag[1] == "skip":
        return {"status": "skipped", "reason": "skip_pair", "explanation": content}

    original = parent.strip()
    if original == content:
        return {"status": "skipped", "reason": "unchanged_child"}
    if not _unchanged_protected_blocks(original, content):
        raise ValueError("Child omitted or changed a protected block")
    old_lines, new_lines = original.splitlines(keepends=True), content.splitlines(keepends=True)
    # Compare lines without their end-of-line tokens so an end insertion does not
    # appear to also replace the previously last line.
    ops = [op for op in SequenceMatcher(None, original.splitlines(), content.splitlines(), autojunk=False).get_opcodes()
           if op[0] != "equal"]
    if len(ops) != 1:
        raise ValueError("Child must change a single localized part of the Basic Prompt")
    op, i, j, a, b = ops[0]
    old = "".join(old_lines[i:j]).rstrip("\r\n")
    new = "".join(new_lines[a:b]).rstrip("\r\n")
    for fragment in (old, new):
        if "\n\n" in fragment or len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", fragment)) > 1:
            raise ValueError("The changed block spans multiple rules")
    if old == original and len(old_lines) > 1:
        raise ValueError("The whole guide was replaced")
    for ex in examples:
        for literal in (str(ex["id"]), ex["question"]):
            if len(literal) >= 6 and literal in new and literal not in old:
                raise ValueError("Example identifier or question copied into the skill")
    # Preserve the native wrapper whitespace (including its trailing newline).
    start = parent.index(original)
    skill = parent[:start] + content + parent[start + len(original):]
    # The training guard formerly used JSON-only model explanations. Keep the
    # approved generation request unchanged and rank its train items from the actual edit.
    return {"status": "generated", "skill": skill, "format": "tagged_full_skill",
            "mutation": new or old, "transfer_cues": {"target": new, "risk": old},
            "edits": [{"index": 1, "op": op, "target": old, "content": new,
                       "parent_line_start": i + 1, "parent_line_count": j-i, "status": "observed_text_diff"}],
            "shown_example_ids": [e["id"] for e in examples]}
