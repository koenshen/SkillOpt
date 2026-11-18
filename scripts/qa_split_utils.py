"""Shared helpers for materializing SearchQA-style QA benchmarks.

SkillOpt's SearchQA adapter consumes items of the form::

    {"id": str, "question": str, "context": str, "answers": [str, ...]}

where ``context`` is a single string with passages separated by ``[DOC]``
(``skillopt/envs/searchqa/rollout.py`` truncates at those markers).  These
helpers turn other QA datasets into that shape and carve deterministic,
difficulty-matched train/val/test splits out of one pool.

The split allocator is *stratified*: each stratum (a difficulty proxy such as
task type or answer kind) contributes the same proportion to every split, to
within one item.  Plain random sampling does not guarantee that -- it only
bounds the deviation in expectation -- and since both source datasets are
ordered by question source, taking a contiguous slice can be badly skewed.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

MAX_CONTEXT_CHARS = 6000


def doc_block(title: str, body: str) -> str:
    """Render one passage in the ``[DOC]`` format the SearchQA rollout expects."""
    title = (title or "").strip()
    body = (body or "").strip()
    if not body:
        return ""
    if title:
        return f"[DOC] {title}\n{body}"
    return f"[DOC]\n{body}"


def join_doc_blocks(blocks: list[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Join passage blocks, dropping any block that would exceed *max_chars*.

    The adapter truncates at ``[DOC]`` boundaries as well, so staying under the
    budget here keeps the cut deterministic and avoids handing it a context it
    would silently shorten.
    """
    out = ""
    for block in blocks:
        if not block:
            continue
        candidate = f"{out}\n\n{block}" if out else block
        if len(candidate) > max_chars:
            continue
        out = candidate
    return out


def _largest_remainder(total: int, weights: list[float]) -> list[int]:
    raw = [total * w for w in weights]
    base = [int(r) for r in raw]
    remainder = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i]), reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base


def stratified_split(
    items: list[dict],
    sizes: dict[str, int],
    key_fn,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split *items* into named splits of exactly the requested sizes.

    Each split's composition follows the pool's stratum composition: a stratum
    with share ``p`` supplies ``p * size`` items to every split, rounded so
    each split lands exactly on its requested size.  Items inside a stratum
    are drawn in a seeded random order, so the draw is reproducible.
    """
    pool_size = len(items)
    if pool_size < sum(sizes.values()):
        raise SystemExit(
            f"pool has {pool_size} items, need {sum(sizes.values())}"
        )

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_stratum[str(key_fn(item))].append(item)

    strata = sorted(by_stratum)
    weights = {s: len(by_stratum[s]) / pool_size for s in strata}
    capacity = {s: len(by_stratum[s]) for s in strata}

    # Desired counts per (split, stratum), then fill each split's row up to its
    # exact size with the largest-remainder stratum that still has spare items.
    # Using floors first and redistributing keeps every split exact without
    # oversubscribing a small stratum (rounding alone can ask for one more item
    # than a tiny stratum has).
    allocation: dict[str, dict[str, int]] = {}
    taken = {s: 0 for s in strata}
    for name, size in sizes.items():
        raw = {s: size * weights[s] for s in strata}
        base = {s: int(raw[s]) for s in strata}
        # Never claim more than what is left after earlier splits.
        for s in strata:
            base[s] = min(base[s], max(0, capacity[s] - taken[s]))
            taken[s] += base[s]
        assignment = dict(base)
        while sum(assignment.values()) < size:
            candidates = [
                s for s in strata
                if taken[s] < capacity[s]
            ]
            if not candidates:
                raise SystemExit(
                    f"cannot fill split {name!r} to {size}: pool exhausted"
                )
            s = max(candidates, key=lambda k: (raw[k] - assignment[k], k))
            assignment[s] += 1
            taken[s] += 1
        allocation[name] = assignment

    out: dict[str, list[dict]] = {name: [] for name in sizes}
    shuffler = random.Random(seed)
    for s in strata:
        bucket = list(by_stratum[s])
        shuffler.shuffle(bucket)
        cursor = 0
        for name in sizes:
            count = allocation[name][s]
            out[name].extend(bucket[cursor:cursor + count])
            cursor += count
    return out


def stratum_report(splits: dict[str, list[dict]], key_fn) -> dict[str, dict[str, float]]:
    """Return each split's stratum shares, for checking the split is balanced."""
    report: dict[str, dict[str, float]] = {}
    for name, items in splits.items():
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            counts[str(key_fn(item))] += 1
        total = max(len(items), 1)
        report[name] = {k: v / total for k, v in sorted(counts.items())}
    return report


def check_splits(splits: dict[str, list[dict]], key_fn, tol: float = 0.02) -> None:
    """Raise if a stratum's share differs between splits by more than *tol*."""
    report = stratum_report(splits, key_fn)
    all_keys = sorted({k for r in report.values() for k in r})
    for key in all_keys:
        shares = [report[name].get(key, 0.0) for name in splits]
        if max(shares) - min(shares) > tol:
            raise SystemExit(
                f"split composition is unbalanced for {key!r}: "
                f"{ {n: round(report[n].get(key, 0.0), 3) for n in splits} }"
            )


def write_splits(
    output_dir: Path,
    splits: dict[str, list[dict]],
    *,
    source: str,
    benchmark: str,
    extra_manifest: dict | None = None,
) -> dict[str, int]:
    """Write ``<output_dir>/<split>/items.json`` and a manifest."""
    output_dir = Path(output_dir).resolve()
    ids: dict[str, str] = {}
    for name, items in splits.items():
        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            item_id = str(item["id"])
            if item_id in ids:
                raise SystemExit(
                    f"duplicate id {item_id!r} in {name} and {ids[item_id]}"
                )
            ids[item_id] = name
        with (split_dir / "items.json").open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    counts = {name: len(items) for name, items in splits.items()}
    manifest = {
        "benchmark": benchmark,
        "source": source,
        "counts": counts,
        "item_fields": ["id", "question", "context", "answers"],
        "split_method": "stratified by difficulty proxy, seeded",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return counts
