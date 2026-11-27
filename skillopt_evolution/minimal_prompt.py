"""Render the approved DE-v1 template and apply its one sentence replacement to the original guide."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata

from skillopt.optimizer.skill import APPENDIX_END, APPENDIX_START, SLOW_UPDATE_END, SLOW_UPDATE_START

TEMPLATE = Path(__file__).with_name("prompts") / "stage2_de_v1_en.txt"
ROLES = {"A": "Prompt 1 (first historical guide)", "B": "Prompt 2 (second historical guide)",
         "X": "Basic Prompt (guide being improved)", "best": "Prompt 3 (current highest-scoring guide)"}


def template_digest():
    return hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


_MARKERS = {SLOW_UPDATE_START: SLOW_UPDATE_END, APPENDIX_START: APPENDIX_END}
_TOKENS = re.compile(r"<!-- (?:SLOW_UPDATE|APPENDIX)_(?:START|END) -->")
_LIST_PREFIX = re.compile(r"^[ \t]*(?:(?:[-*+] |\d+[.)] )[ \t]*)?")
# Inline code (including punctuation in examples) is one unit. Full stops in
# abbreviations are not sentence terminators. Offsets refer to the original.
_SENTENCE = re.compile(r"`[^`]*`|[.!?]+(?:[’”\"']|\*\*)?(?=\s|$)")
_ABBREVIATION = re.compile(r"(?:\b(?:e\.g|i\.e|Mr|Mrs|Ms|Dr|Prof|vs|etc)|\b[A-Z])\.$", re.I)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    editable: bool


def _sentences(line):
    start = 0
    for m in _SENTENCE.finditer(line):
        if line[m.start()] == "`" or _ABBREVIATION.search(line[:m.end()]):
            continue
        yield start, m.end()
        start = m.end()
    if start < len(line):
        yield start, len(line)


def guide_spans(guide):
    """Return reference-only blocks/headers and exact editable source offsets."""
    protected = []
    opened = None
    for m in _TOKENS.finditer(guide):
        token = m.group()
        if token in _MARKERS and opened is None:
            opened = (m.start(), _MARKERS[token])
        elif opened and token == opened[1]:
            protected.append((opened[0], m.end()))
            opened = None
        else:
            raise ValueError("Unbalanced protected markers in the Basic Prompt")
    if opened:
        raise ValueError("Unclosed protected block in the Basic Prompt")
    spans = []
    offset = 0
    for lo, hi in [*protected, (len(guide), len(guide))]:
        for line in guide[offset:lo].splitlines(keepends=True):
            if line.lstrip().startswith("#"):
                spans.append(Span(offset, offset + len(line.rstrip()), False))
            else:
                prefix = _LIST_PREFIX.match(line).end()
                for a, b in _sentences(line[prefix:]):
                    chunk = line[prefix+a:prefix+b]
                    if chunk.strip():
                        spans.append(Span(offset+prefix+a+len(chunk)-len(chunk.lstrip()),
                                          offset+prefix+a+len(chunk.rstrip()), True))
            offset += len(line)
        if lo < hi:
            spans.append(Span(lo, hi, False))
        offset = hi
    counts = Counter(guide[s.start:s.end] for s in spans if s.editable)
    return [Span(s.start, s.end, s.editable and counts[guide[s.start:s.end]] == 1) for s in spans]


def annotated_guide(guide):
    return "\n".join(("[EDITABLE] " if s.editable else "[READ ONLY] ") + guide[s.start:s.end]
                     for s in guide_spans(guide))


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
    subs = {"prompt0": annotated_guide(records["X"]["skill"]), "prompt1": records["A"]["skill"],
            "prompt2": records["B"]["skill"], "prompt3": records["best"]["skill"],
            "answer_records": "\n".join(evidence)}
    # Single pass: placeholder-like text in real documents is literal data.
    return re.sub(r"<(prompt[0-3]|answer_records)>", lambda m: subs[m[1]], TEMPLATE.read_text(encoding="utf-8"))


_RESPONSE = re.compile(r"<old>(?P<old>.*?)</old>\s*(?:→|->)?\s*<new>(?P<new>.*?)</new>", re.S)
_CONTROL = re.compile(r"</?(?:old|new|skip|prompt)>|\[(?:EDITABLE|READ ONLY)\]")


def _words(text):
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def _occurrences(tokens, phrase):
    return sum(tokens[i:i+len(phrase)] == phrase for i in range(len(tokens)-len(phrase)+1))


def _copied_answers(old, new, examples):
    before, after = _words(old), _words(new)
    copied = []
    for ex in examples:
        for role, row in ex["answers"].items():
            answer = row["answer"]
            phrase = _words(answer)
            # TriviaQA EM permits articles to differ (e.g. "The Barretts").
            if len(phrase) > 1 and phrase[0] in ("a", "an", "the"):
                phrase = phrase[1:]
            if phrase and _occurrences(after, phrase) > _occurrences(before, phrase):
                copied.append({"id": ex["id"], "role": role, "answer": answer})
    return copied


def parse_reply(raw, parent, examples):
    """Reconstruct locally: everything outside the selected sentence is byte-for-byte identical."""
    reply = raw.strip()
    fenced = re.fullmatch(r"```(?:text|xml)?[ \t]*\r?\n(.*?)\r?\n```", reply, re.S)
    if fenced:
        reply = fenced[1].strip()
    reply = reply.removeprefix("Final edit:").strip()
    skipped = re.fullmatch(r"<skip>(.*?)</skip>", reply, re.S)
    if skipped and skipped[1].strip() and not _CONTROL.search(skipped[1]):
        return {"status": "skipped", "reason": "skip_pair", "explanation": skipped[1].strip()}
    pair = _RESPONSE.fullmatch(reply)
    if not pair and not _CONTROL.search(reply) and reply.count("→") == 1:
        pair = re.fullmatch(r"(?P<old>[^\n]+?)\s*→\s*(?P<new>[^\n]+)", reply)
    if not pair:
        raise ValueError("Expected only <old>original</old> → <new>replacement</new> or <skip>reason</skip>")
    old, new = (pair[k].strip() for k in ("old", "new"))
    if (not old or not new or _CONTROL.search(old) or _CONTROL.search(new)
            or _TOKENS.search(new) or "\n" in new or "\r" in new):
        raise ValueError("Expected exactly one nonempty, single-sentence replacement")
    matches = [s for s in guide_spans(parent) if s.editable and parent[s.start:s.end] == old]
    if len(matches) != 1:
        raise ValueError("Original sentence is not a unique [EDITABLE] sentence")
    if old == new:
        return {"status": "skipped", "reason": "unchanged_child"}
    if len([1 for a,b in _sentences(new) if new[a:b].strip()]) != 1:
        raise ValueError("Replacement contains multiple sentences")
    for ex in examples:
        for literal in (str(ex["id"]), ex["question"]):
            if len(literal) >= 6 and literal in new and literal not in old:
                raise ValueError("Example identifier or question copied into the skill")
    copied = _copied_answers(old, new, examples)
    if copied:
        return {"status": "skipped", "reason": "copied_displayed_answer",
                "matches": copied, "target": old, "content": new}
    span = matches[0]
    return {"status": "generated", "skill": parent[:span.start] + new + parent[span.end:],
            "format": "sentence_replacement", "mutation": new, "transfer_cues": {"target": new, "risk": old},
            "edits": [{"index": 1, "op": "replace", "target": old, "content": new,
                       "parent_line_start": 1 + parent.count("\n", 0, span.start),
                       "start_offset": span.start, "end_offset": span.end, "status": "applied_exact_replace"}],
            "shown_example_ids": [e["id"] for e in examples]}
