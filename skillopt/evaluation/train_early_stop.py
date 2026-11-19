"""Training-level early stop for the SkillOpt loop (off by default).

Selection early stop (``early_stop.py``) only shortens the validation of one
candidate and never changes a gate decision.  This module is the aggressive
companion: it ends the whole training run once the recent candidates show no
real progress over the current skill, so the remaining steps are not paid for.

Per step we compare the candidate with the current skill item by item on the
selection set:

    d = sum(cand_i - cur_i)             net items gained (+) or lost (-)
    u = sum(|cand_i - cur_i|)           discordant items
    z = d / sqrt(u)                     standardised progress (0 if u == 0)

and smooth it two ways:

    ema   = lam * z + (1 - lam) * ema            (reported only)
    cusum = max(0, cusum + z - delta)            evidence of progress above delta
    flat  = flat + 1 if cusum == 0 else 0        consecutive steps without it

Training stops after the step where

    step >= max(min_steps, ceil(rho * total_steps))  and  flat >= k

``rho`` is the earliest point training may stop; before it the run behaves
exactly as without this module.  Steps with no evaluated candidate (no patches,
no rewrite, cache hit) count as z = 0, i.e. no progress.

This is not exact: a later step might still have produced a better skill.  It
trades that risk for tokens, and the best skill seen so far is what the final
evaluation uses, as in a full run.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass


@dataclass
class TrainEarlyStopConfig:
    rho: float = 0.75
    k: int = 3
    delta: float = 0.5
    lam: float = 0.6
    min_steps: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"evaluation.train_early_stop_rho must be in [0, 1], got {self.rho}")
        if self.k < 1:
            raise ValueError(f"evaluation.train_early_stop_k must be >= 1, got {self.k}")
        if self.delta < 0:
            raise ValueError(f"evaluation.train_early_stop_delta must be >= 0, got {self.delta}")
        if not 0.0 < self.lam <= 1.0:
            raise ValueError(f"evaluation.train_early_stop_lambda must be in (0, 1], got {self.lam}")
        if self.min_steps < 0:
            raise ValueError(f"evaluation.train_early_stop_min_steps must be >= 0, got {self.min_steps}")

    @classmethod
    def from_cfg(cls, cfg: dict) -> "TrainEarlyStopConfig":
        def pick(key, default):
            val = cfg.get(key)
            return default if val is None or val == "" else val

        return cls(
            rho=float(pick("train_early_stop_rho", 0.75)),
            k=int(pick("train_early_stop_k", 3)),
            delta=float(pick("train_early_stop_delta", 0.5)),
            lam=float(pick("train_early_stop_lambda", 0.6)),
            min_steps=int(pick("train_early_stop_min_steps", 10)),
        )


@dataclass
class TrainEarlyStopState:
    ema: float = 0.0
    cusum: float = 0.0
    flat_steps: int = 0
    n_obs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrainEarlyStopState":
        d = d or {}
        return cls(
            ema=float(d.get("ema", 0.0) or 0.0),
            cusum=float(d.get("cusum", 0.0) or 0.0),
            flat_steps=int(d.get("flat_steps", 0) or 0),
            n_obs=int(d.get("n_obs", 0) or 0),
        )


def item_scores(results: list | None) -> dict[str, float]:
    """``{id: hard}`` with hard clipped to [0, 1]; rows without an id are ignored."""
    out: dict[str, float] = {}
    for r in results or []:
        if not isinstance(r, dict) or "id" not in r:
            continue
        try:
            h = float(r.get("hard", 0) or 0)
        except (TypeError, ValueError):
            h = 0.0
        out[str(r["id"])] = min(1.0, max(0.0, h))
    return out


def load_item_scores(path: str | None) -> dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except (ValueError, RecursionError, TypeError):
                pass
    return item_scores(rows)


def paired_progress(cand: dict[str, float], cur: dict[str, float]) -> dict:
    """Paired comparison of candidate vs current on their common items."""
    common = [i for i in cand if i in cur]
    d = sum(cand[i] - cur[i] for i in common)
    u = sum(abs(cand[i] - cur[i]) for i in common)
    z = d / math.sqrt(u) if u > 1e-12 else 0.0
    return {"z": z, "d": d, "u": u, "n": len(common)}


def threshold_step(total_steps: int, cfg: TrainEarlyStopConfig) -> int:
    """Earliest step after which training may stop."""
    return max(int(cfg.min_steps), int(math.ceil(cfg.rho * total_steps)))


def observe(state: TrainEarlyStopState, z: float, cfg: TrainEarlyStopConfig) -> TrainEarlyStopState:
    """Fold one step's progress ``z`` into the state (in place) and return it."""
    state.ema = cfg.lam * z + (1.0 - cfg.lam) * state.ema
    state.cusum = max(0.0, state.cusum + z - cfg.delta)
    state.flat_steps = state.flat_steps + 1 if state.cusum <= 0.0 else 0
    state.n_obs += 1
    return state


def should_stop(
    state: TrainEarlyStopState,
    global_step: int,
    total_steps: int,
    cfg: TrainEarlyStopConfig,
) -> bool:
    if global_step >= total_steps:
        return False  # the run ends here anyway; nothing to save
    return global_step >= threshold_step(total_steps, cfg) and state.flat_steps >= cfg.k
