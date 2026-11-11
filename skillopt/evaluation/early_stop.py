"""Exact, lineage-ordered early stopping for selection-set validation.

The gate accepts a candidate only if its hard score is strictly greater than
the current skill's score.  While evaluating a candidate item by item, the
best score it can still reach is

    upper = (hard_sum_so_far + n_not_yet_evaluated) / n_total

because every item scores at most 1.  Once ``upper <= current_score`` the
candidate cannot be accepted whatever the remaining items return, so the
remaining items can be skipped without changing the gate decision.

The order in which items are evaluated does not affect correctness, only how
early the stop happens.  We order items by their mean hard score over all
previously evaluated skill versions (lowest first): items that earlier
versions usually failed are the most likely to stay failed, so they push
``upper`` down fastest.

Accepted candidates are always evaluated on every item, so their score is
exact.  Rejected candidates that stopped early only have an upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EarlyStopOutcome:
    results: list[dict]          # results of the evaluated items only
    n_total: int
    n_evaluated: int
    stopped: bool                # True if items were skipped
    upper_bound: float           # max achievable hard score
    lower_bound: float           # hard score if every skipped item fails
    batches: list[dict] = field(default_factory=list)


def _hard(r: dict) -> float:
    return min(1.0, max(0.0, float(r.get("hard", 0) or 0)))


def lineage_order(item_ids: list[str], history: dict[str, list[float]]) -> list[str]:
    """Sort item ids by mean historical hard score, lowest first.

    Items never observed are treated as 0.  Ties keep the original order, so
    the result is deterministic.
    """
    def key(i: str) -> float:
        obs = history.get(i)
        return sum(obs) / len(obs) if obs else 0.0

    return sorted(item_ids, key=key)


def update_history(history: dict[str, list[float]], results: list[dict]) -> None:
    for r in results:
        if "id" in r:
            history.setdefault(str(r["id"]), []).append(_hard(r))


def evaluate_with_early_stop(
    items: list[dict],
    history: dict[str, list[float]],
    run_items: Callable[[list[dict]], list[dict]],
    current_score: float,
    batch_size: int,
    log: Callable[[str], None] | None = None,
) -> EarlyStopOutcome:
    """Evaluate ``items`` batch by batch in lineage order, stopping once the
    candidate provably cannot beat ``current_score``.

    ``run_items(prefix)`` must evaluate every item in ``prefix`` that has not
    been evaluated yet and return results for (at least) the whole prefix;
    a resume-aware rollout writing to one directory does exactly that.
    """
    n = len(items)
    batch_size = max(1, int(batch_size))
    by_id = {str(it["id"]): it for it in items}
    order = lineage_order([str(it["id"]) for it in items], history)

    done = 0
    evaluated: dict[str, dict] = {}
    batches: list[dict] = []
    upper = 1.0
    while done < n:
        done = min(n, done + batch_size)
        prefix_ids = order[:done]
        wanted = set(prefix_ids)
        out = run_items([by_id[i] for i in prefix_ids])
        evaluated = {str(r["id"]): r for r in out if str(r.get("id")) in wanted}
        hard_sum = sum(_hard(r) for r in evaluated.values())
        upper = (hard_sum + (n - len(evaluated))) / n
        batches.append({"n_evaluated": len(evaluated), "hard_sum": hard_sum, "upper": upper})
        if done < n and upper <= current_score:
            if log:
                log(
                    f"early stop after {len(evaluated)}/{n} items: "
                    f"correct={hard_sum:g} upper={upper:.4f} <= current={current_score:.4f}"
                )
            results = [evaluated[i] for i in prefix_ids if i in evaluated]
            return EarlyStopOutcome(
                results=results, n_total=n, n_evaluated=len(evaluated), stopped=True,
                upper_bound=upper, lower_bound=hard_sum / n, batches=batches,
            )

    results = [evaluated[i] for i in order if i in evaluated]
    hard_sum = sum(_hard(r) for r in results)
    return EarlyStopOutcome(
        results=results, n_total=n, n_evaluated=len(results), stopped=False,
        upper_bound=(hard_sum + (n - len(results))) / n, lower_bound=hard_sum / n,
        batches=batches,
    )
