"""Tests for training-level early stop (evaluation.train_early_stop)."""
import json
import math
import os

import pytest

from skillopt.evaluation.train_early_stop import (
    TrainEarlyStopConfig,
    TrainEarlyStopState,
    item_scores,
    observe,
    paired_progress,
    should_stop,
    threshold_step,
)


# ── Pure functions ───────────────────────────────────────────────────────


def test_paired_progress_counts_only_discordant_items():
    cur = {"a": 1, "b": 0, "c": 1, "d": 0}
    cand = {"a": 1, "b": 1, "c": 0, "d": 1}
    p = paired_progress(cand, cur)
    assert p["d"] == 1 and p["u"] == 3 and p["n"] == 4
    assert p["z"] == pytest.approx(1 / math.sqrt(3))


def test_paired_progress_uses_common_items_only():
    cur = {"a": 1, "b": 0, "c": 0}
    cand = {"b": 1, "x": 0}  # partial eval, plus an unknown id
    p = paired_progress(cand, cur)
    assert p["n"] == 1 and p["d"] == 1 and p["u"] == 1 and p["z"] == 1.0


def test_paired_progress_identical_is_zero():
    s = {"a": 1, "b": 0}
    assert paired_progress(s, s)["z"] == 0.0


def test_item_scores_clips_and_skips_rows_without_id():
    rows = [{"id": 1, "hard": 2}, {"id": "b", "hard": -1}, {"hard": 1}, None]
    assert item_scores(rows) == {"1": 1.0, "b": 0.0}


def test_threshold_step_respects_rho_and_min_steps():
    cfg = TrainEarlyStopConfig(rho=0.75, min_steps=10)
    assert threshold_step(40, cfg) == 30
    assert threshold_step(160, cfg) == 120
    assert threshold_step(84, cfg) == 63
    assert threshold_step(8, cfg) == 10           # min_steps wins on short runs
    assert threshold_step(40, TrainEarlyStopConfig(rho=0.6)) == 24  # mid-epoch


def test_observe_cusum_and_flat_counter():
    cfg = TrainEarlyStopConfig(delta=0.5, lam=0.5, k=2)
    st = TrainEarlyStopState()
    observe(st, 2.0, cfg)          # strong progress: cusum 1.5
    assert st.cusum == pytest.approx(1.5) and st.flat_steps == 0
    observe(st, -2.0, cfg)         # wipes it out
    assert st.cusum == 0.0 and st.flat_steps == 1
    observe(st, 0.3, cfg)          # below delta: still no progress
    assert st.cusum == 0.0 and st.flat_steps == 2
    observe(st, 1.0, cfg)          # progress above delta resets the counter
    assert st.cusum == pytest.approx(0.5) and st.flat_steps == 0
    assert st.n_obs == 4


def test_should_stop_needs_threshold_and_k():
    cfg = TrainEarlyStopConfig(rho=0.5, k=3, min_steps=0)
    flat = TrainEarlyStopState(flat_steps=3)
    assert not should_stop(flat, 19, 40, cfg)   # before rho*N
    assert should_stop(flat, 20, 40, cfg)
    assert not should_stop(TrainEarlyStopState(flat_steps=2), 25, 40, cfg)
    assert not should_stop(flat, 40, 40, cfg)   # last step: nothing to save


def test_state_roundtrip():
    st = TrainEarlyStopState(ema=-0.4, cusum=1.25, flat_steps=2, n_obs=7)
    assert TrainEarlyStopState.from_dict(st.to_dict()) == st
    assert TrainEarlyStopState.from_dict(None) == TrainEarlyStopState()


@pytest.mark.parametrize("kw", [{"rho": 1.5}, {"k": 0}, {"delta": -1}, {"lam": 0}, {"min_steps": -1}])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        TrainEarlyStopConfig(**kw)


def test_config_from_cfg_defaults_and_overrides():
    assert TrainEarlyStopConfig.from_cfg({}) == TrainEarlyStopConfig()
    c = TrainEarlyStopConfig.from_cfg({"train_early_stop_rho": "0.6", "train_early_stop_k": 5})
    assert c.rho == 0.6 and c.k == 5


def test_config_keys_flatten_from_yaml():
    from skillopt.config import _FLATTEN_MAP

    for key in ("", "_rho", "_k", "_delta", "_lambda", "_min_steps"):
        assert _FLATTEN_MAP[f"evaluation.train_early_stop{key}"] == f"train_early_stop{key}"


# ── End-to-end trainer runs with a fake adapter and stubbed optimizer ────


N_ITEMS = 20


class _FakeAdapter:
    """Selection items q0..q19.  Version v of the skill solves the first
    min(10 + v, 14) items, so steps 1-4 improve and every later candidate is
    identical in behaviour (no progress)."""

    def __init__(self):
        self.rollout_calls = []

    def setup(self, cfg):
        pass

    def get_dataloader(self):
        return None

    def requires_ray(self):
        return False

    def get_task_types(self):
        return ["qa"]

    def build_train_env(self, batch_size, seed, **kw):
        return [{"id": f"t{seed}_{i}"} for i in range(2)]

    def build_eval_env(self, env_num, split, seed, **kw):
        return [{"id": f"q{i}"} for i in range(N_ITEMS)]

    @staticmethod
    def _version(skill):
        return skill.count("RULE")

    def rollout(self, env, skill, out_dir, **kw):
        self.rollout_calls.append(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        solved = min(10 + self._version(skill), 14)
        rows = []
        for it in env:
            iid = str(it["id"])
            idx = int(iid[1:]) if iid.startswith("q") else 0
            hard = 1 if (not iid.startswith("q") or idx < solved) else 0
            rows.append({"id": iid, "hard": hard, "soft": float(hard)})
        with open(os.path.join(out_dir, "results.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return rows

    def reflect(self, rollout_results, skill, batch_dir, **kw):
        return [{"source_type": "failure", "batch_size": 1, "patch": {"edits": [{"op": "add"}]}}]


def _run(tmp_path, monkeypatch, **overrides):
    import skillopt.engine.trainer as tm

    for name in (
        "configure_azure_openai", "set_optimizer_backend", "set_target_backend",
        "set_optimizer_deployment", "set_target_deployment", "configure_codex_exec_from_config",
        "configure_claude_code_exec", "configure_cursor_exec", "configure_copilot_exec",
        "configure_copilot_chat", "configure_qwen_chat", "configure_minimax_chat",
        "_configure_trace_to_optimizer_gates", "set_reasoning_effort",
    ):
        monkeypatch.setattr(tm, name, lambda *a, **k: None)
    monkeypatch.setattr(tm, "_resolve_role_backends", lambda b, o, t: ("x", "x"))
    monkeypatch.setattr(tm, "merge_patches", lambda skill, f, s, **k: {"edits": [{"op": "add"}]})
    monkeypatch.setattr(tm, "rank_and_select", lambda skill, patch, **k: patch)
    counter = {"n": 0}

    def apply(skill, patch):
        counter["n"] += 1
        return skill + f"\nRULE {counter['n']}", [{"status": "applied"}]

    monkeypatch.setattr(tm, "apply_patch_with_report", apply)

    cfg = {
        "out_root": str(tmp_path), "optimizer_model": "o", "target_model": "t",
        "skill_init": str(tmp_path / "missing.md"),
        "batch_size": 2, "num_epochs": 4, "accumulation": 1, "seed": 0,
        "merge_batch_size": 2, "train_size": 10, "edit_budget": 4,
        "analyst_workers": 1, "sel_env_num": 0, "test_env_num": 0,
        "eval_test": False, "use_slow_update": False, "use_meta_skill": False,
    }
    cfg.update(overrides)
    adapter = _FakeAdapter()
    tm.ReflACTTrainer(cfg, adapter).train()
    with open(tmp_path / "history.json") as f:
        history = json.load(f)
    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)
    return history, summary, adapter


def test_default_off_runs_all_steps_and_adds_nothing(tmp_path, monkeypatch):
    history, summary, _ = _run(tmp_path, monkeypatch)
    assert len(history) == 20                       # 4 epochs x 5 steps
    assert all("train_early_stop" not in r for r in history)
    assert "train_early_stop" not in summary
    with open(tmp_path / "runtime_state.json") as f:
        assert "train_early_stopped_at" not in json.load(f)


def test_off_and_on_agree_until_the_stop(tmp_path, monkeypatch):
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    h_off, s_off, _ = _run(off_dir, monkeypatch)
    h_on, s_on, _ = _run(
        on_dir, monkeypatch, train_early_stop=True,
        train_early_stop_rho=0.5, train_early_stop_k=3, train_early_stop_min_steps=0,
    )
    stop = s_on["train_early_stop"]["stop_step"]
    assert stop is not None and 10 <= stop < 20
    assert len(h_on) == stop
    keys = ("step", "action", "selection_hard", "current_score", "best_score", "best_step")
    for a, b in zip(h_off[:stop], h_on):
        assert {k: a.get(k) for k in keys} == {k: b.get(k) for k in keys}
    assert s_on["best_step"] == s_off["best_step"] == 4   # best skill kept
    assert s_on["best_selection_hard"] == s_off["best_selection_hard"]


def test_on_stops_mid_epoch_at_rho(tmp_path, monkeypatch):
    # rho=0.6 of 20 steps -> threshold 12, which is inside epoch 3 (steps 11-15).
    history, summary, _ = _run(
        tmp_path, monkeypatch, train_early_stop=True,
        train_early_stop_rho=0.6, train_early_stop_k=3, train_early_stop_min_steps=0,
    )
    info = summary["train_early_stop"]
    assert info["threshold_step"] == 12
    assert info["stop_step"] == 12 and len(history) == 12
    assert history[-1]["epoch"] == 3 and history[-1]["step_in_epoch"] == 1
    assert history[0]["train_early_stop"]["z"] > 0           # step 1 improves
    assert history[-1]["train_early_stop"]["state"]["flat_steps"] >= 3


def test_on_skips_end_of_epoch_slow_update_after_stop(tmp_path, monkeypatch):
    import skillopt.engine.trainer as tm

    calls = []
    monkeypatch.setattr(tm, "run_slow_update", lambda *a, **k: calls.append(1) or None)
    history, summary, _ = _run(
        tmp_path, monkeypatch, train_early_stop=True, use_slow_update=True,
        train_early_stop_rho=0.6, train_early_stop_k=3, train_early_stop_min_steps=0,
    )
    assert summary["train_early_stop"]["stop_step"] == 12
    assert not os.path.exists(tmp_path / "slow_update" / "epoch_03")
    assert os.path.exists(tmp_path / "slow_update" / "epoch_02")


def test_never_stops_before_min_steps(tmp_path, monkeypatch):
    history, summary, _ = _run(
        tmp_path, monkeypatch, train_early_stop=True,
        train_early_stop_rho=0.0, train_early_stop_k=1, train_early_stop_min_steps=8,
    )
    assert summary["train_early_stop"]["stop_step"] == 8


def test_no_stop_when_rho_is_one(tmp_path, monkeypatch):
    history, summary, _ = _run(
        tmp_path, monkeypatch, train_early_stop=True, train_early_stop_rho=1.0,
    )
    assert len(history) == 20
    assert summary["train_early_stop"]["stopped"] is False


def test_resume_after_stop_does_not_train_further(tmp_path, monkeypatch):
    kw = dict(train_early_stop=True, train_early_stop_rho=0.6,
              train_early_stop_k=3, train_early_stop_min_steps=0)
    h1, s1, _ = _run(tmp_path, monkeypatch, **kw)
    h2, s2, adapter = _run(tmp_path, monkeypatch, **kw)
    assert len(h2) == len(h1) == 12
    assert s2["train_early_stop"]["stop_step"] == 12
    assert not any("/steps/" in c for c in adapter.rollout_calls)


def test_resume_mid_run_restores_state(tmp_path, monkeypatch):
    """Interrupt after step 9, resume, and land on the same stop step."""
    import skillopt.engine.trainer as tm

    kw = dict(train_early_stop=True, train_early_stop_rho=0.6,
              train_early_stop_k=3, train_early_stop_min_steps=0)
    full_dir = tmp_path / "full"
    _, s_full, _ = _run(full_dir, monkeypatch, **kw)

    part_dir = tmp_path / "part"
    real_save = tm._save_runtime_state

    class _Interrupt(Exception):
        pass

    def save_then_stop(out_root, state):
        # Interrupt only once step 9 is fully persisted (history + state).
        real_save(out_root, state)
        if state.get("last_completed_step") == 9:
            raise _Interrupt

    monkeypatch.setattr(tm, "_save_runtime_state", save_then_stop)
    with pytest.raises(_Interrupt):
        _run(part_dir, monkeypatch, **kw)
    monkeypatch.setattr(tm, "_save_runtime_state", real_save)
    h2, s2, _ = _run(part_dir, monkeypatch, **kw)
    assert s2["train_early_stop"]["stop_step"] == s_full["train_early_stop"]["stop_step"]
    assert len(h2) == 12
