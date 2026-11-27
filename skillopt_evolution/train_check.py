"""Check one localized edit on separate training items before it occupies a dev slot."""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from pathlib import Path

from .evaluate import evaluate_candidate
from .usage import read_json, save_json


def words(text):
    return re.findall(r"[^\W_]+", str(text).casefold())


def normal(text):
    return " ".join(words(text))


def named_phrases(text):
    # Multi-word proper names shared with a displayed dev clue would make a
    # poor transfer example. Answer-alias and duplicate-question checks below
    # also apply. This is a surface filter rather than an entity linker.
    return {normal(x) for x in re.findall(r"\b[A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+)+", text)}


def answers(item):
    return {normal(s) for s in item.get("answers", []) if normal(s)}


class TrainCheck:
    def __init__(self, train, dev):
        self.dev = {str(q["id"]): q for q in dev}
        dev_text = {normal(q["question"]) for q in dev}
        self.train = {}
        self.terms = {}
        frequencies = Counter()
        seen = set()
        for item in train:
            k, text = str(item["id"]), normal(item["question"])
            if k in self.dev or text in dev_text:
                continue
            if k in self.train:
                raise ValueError(f"Duplicate train ID {k}")
            if text in seen:
                continue
            seen.add(text)
            self.train[k] = item
            terms = set(words(text))
            self.terms[k] = terms
            frequencies.update(terms)
        self.weights = {w: math.log((1 + len(self.train)) / (1 + n)) for w, n in frequencies.items()}

    def check(self, *, adapter, parent, proposal, selection, out_dir, seed):
        root = Path(out_dir)
        receipt = root / "result.json"
        if receipt.exists():
            return read_json(receipt)
        # A training question must be different from every dev item. Also
        # exclude the known answer aliases of the examples used to propose this edit.
        used = [self.dev[k] for k in proposal["shown_example_ids"]]
        excluded_answers = set().union(*(answers(it) for it in used))
        dev_entities = set().union(*(named_phrases(it["question"]) for it in used))
        pool = [k for k, it in self.train.items()
                if not answers(it) & excluded_answers and not named_phrases(it["question"]) & dev_entities]
        rng = random.Random(seed)
        rng.shuffle(pool)
        def ranked(kind):
            # Prefer overlap with the reusable cues; use question wording as a
            # secondary signal. Neither the child nor its train scores pick examples.
            cue = set(words(proposal["transfer_cues"][kind]))
            questions = set()
            for k in selection[kind + "_ids"]:
                if k in proposal["shown_example_ids"]:
                    questions.update(words(self.dev[k]["question"]))
            def score(k):
                t = self.terms[k]
                return (sum(self.weights[w] for w in sorted(cue & t)) +
                        .5 * sum(self.weights[w] for w in sorted(questions & t))) / math.sqrt(max(1, len(t)))
            return sorted(pool, key=score, reverse=True)

        def score(ids, skill, path, reference=None):
            items = [self.train[k] for k in ids]
            if reference is None:
                # The API requires a reference score for gate comparisons.
                # require_complete evaluates every ID regardless of this placeholder.
                reference = {k: {"id": k, "hard": 0, "agent_ok": True} for k in ids}
            return evaluate_candidate(adapter=adapter, dev_items=items, order=ids, skill=skill,
                parent_rows={k: reference[k] for k in ids}, cached_rows={}, out_dir=path,
                batch_size=adapter.workers, require_complete=True, usage_stage="train_check")["rows"]

        target = ranked("target")[:4]
        if len(target) < 4:
            raise ValueError("Insufficient eligible train questions")
        used_ids = set(target)
        before = score(target, parent["skill"], root / "target_parent")
        groups = {"target": target}
        for kind, ordering in [("related_protection", ranked("risk")), ("general_protection", pool)]:
            pool_ids = [k for k in ordering if k not in used_ids]
            keep = []
            offset = 0
            while len(keep) < 4:
                more = pool_ids[offset:offset + 4 - len(keep)]
                if not more:
                    raise ValueError(f"Could not find four train items answered correctly by X for {kind}")
                used_ids.update(more)
                rr = score(more, parent["skill"], root / kind / f"parent_{offset:04d}")
                before.update(rr)
                keep.extend(k for k in more if rr[k]["hard"])
                offset += len(more)
            groups[kind] = keep
        ids = [k for group in groups.values() for k in group]
        save_json(root / "panel.json", {"groups": groups, "questions": {k: self.train[k]["question"] for k in ids},
                                        "parent": parent["id"], "seed": seed, "cues": proposal["transfer_cues"]})
        after = score(ids, proposal["skill"], root / "child", before)
        lost = [k for k in ids if before[k]["hard"] and not after[k]["hard"]]
        gained = [k for k in ids if not before[k]["hard"] and after[k]["hard"]]
        confirmed = []
        if lost:
            fresh_x = score(lost, parent["skill"], root / "repeat_parent")
            fresh_y = score(lost, proposal["skill"], root / "repeat_child", fresh_x)
            confirmed = [k for k in lost if fresh_x[k]["hard"] and not fresh_y[k]["hard"]]
        result = {"passed": not confirmed, "n": len(ids), "groups": groups, "fixes": gained,
                  "first_regressions": lost, "confirmed_regressions": confirmed}
        save_json(receipt, result)
        return result
