# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

## 261003 self command - install venv
```bash
uv venv --python 3.11.15
source .venv/bin/activate
uv pip install -r requirements.txt

export OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS=131072
export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS=131072
export SKILLOPT_WORKERS=24

# default workers: searchqa=24,officeqa=4,livemath=64

export EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1/embeddings"
export EMBEDDING_API_KEY="sk-xxx"
export EMBEDDING_MODEL="BAAI/bge-m3"

export OPENAI_COMPATIBLE_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-xxx"
export OPENAI_COMPATIBLE_MODEL="deepseek-v4-1-flash/huoshan/huoshan"

export OPENAI_COMPATIBLE_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-xxx"
export OPENAI_COMPATIBLE_MODEL="glm-5.3-flash/tencent/sfb"

export OPENAI_COMPATIBLE_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-xxx"
export OPENAI_COMPATIBLE_MODEL="gemini-3.8-flash/vertex_L/qwa"

export OPENAI_COMPATIBLE_BASE_URL="http://106.75.235.194:6077/v1"
export OPENAI_COMPATIBLE_API_KEY="empty"
export OPENAI_COMPATIBLE_MODEL="Qwen3.8-27B"

export OPENAI_COMPATIBLE_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-xxx"
export OPENAI_COMPATIBLE_MODEL="gpt-5.4-nano"
```

## 261003 self command - data download
```bash
uv pip install -e ".[searchqa]"
python scripts/materialize_searchqa.py

uv pip install -e ".[alfworld]"
uv pip install "alfworld[full]"
alfworld-download

mkdir -p data/raw/livemathematicianbench
hf download LiveMathematicianBench/LiveMathematicianBench \
  --repo-type dataset \
  --local-dir data/raw/livemathematicianbench
python scripts/materialize_livemathematicianbench.py

mkdir -p data/raw/spreadsheetbench
hf download KAKA22/SpreadsheetBench \
  --repo-type dataset \
  --local-dir data/raw/spreadsheetbench
tar -xzf \
  data/raw/spreadsheetbench/spreadsheetbench_verified_400.tar.gz \
  -C data
python scripts/materialize_spreadsheetbench.py

hf auth login
mkdir -p data/raw/officeqa
hf download databricks/officeqa \
  --repo-type dataset \
  --local-dir data/raw/officeqa

# BIRD (text-to-SQL): ID split is in data/bird_id_split; prompts/DBs live in bird-text-to-sql-260820 (read-only)
BIRD_ROOT=/mnt/sdd3/xuancheng/code/github/bird-text-to-sql-260820
python scripts/materialize_bird.py \
  --dev-jsonl $BIRD_ROOT/datas/dev_prompt.generated.jsonl \
  --mini-dev-jsonl $BIRD_ROOT/datas/mini_dev_prompt.generated.jsonl
```

## 261003 self command - train and test
```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-1-flash/huoshan/huoshan \
    model.target=deepseek-v4-1-flash/huoshan/huoshan
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=glm-5.3-flash/tencent/sfb \
    model.target=glm-5.3-flash/tencent/sfb
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=Qwen3.8-27B \
    model.target=Qwen3.8-27B
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=gpt-5.4-nano \
    model.target=gpt-5.4-nano


python scripts/train.py \
  --config configs/livemathematicianbench/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-1-flash/huoshan/huoshan \
    model.target=deepseek-v4-1-flash/huoshan/huoshan
python scripts/train.py \
  --config configs/livemathematicianbench/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=glm-5.3-flash/tencent/sfb \
    model.target=glm-5.3-flash/tencent/sfb
python scripts/train.py \
  --config configs/livemathematicianbench/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=Qwen3.8-27B \
    model.target=Qwen3.8-27B
python scripts/train.py \
  --config configs/livemathematicianbench/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=gpt-5.4-nano \
    model.target=gpt-5.4-nano


python scripts/train.py \
  --config configs/officeqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    env.data_dirs=data/raw/officeqa/treasury_bulletins_parsed/transformed \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-1-flash/huoshan/huoshan \
    model.target=deepseek-v4-1-flash/huoshan/huoshan
python scripts/train.py \
  --config configs/officeqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    env.data_dirs=data/raw/officeqa/treasury_bulletins_parsed/transformed \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=glm-5.3-flash/tencent/sfb \
    model.target=glm-5.3-flash/tencent/sfb
python scripts/train.py \
  --config configs/officeqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    env.data_dirs=data/raw/officeqa/treasury_bulletins_parsed/transformed \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=Qwen3.8-27B \
    model.target=Qwen3.8-27B
python scripts/train.py \
  --config configs/officeqa/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    env.data_dirs=data/raw/officeqa/treasury_bulletins_parsed/transformed \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=gpt-5.4-nano \
    model.target=gpt-5.4-nano


export ALFWORLD_DATA="$HOME/.cache/alfworld"
python scripts/train.py \
  --config configs/alfworld/default.yaml \
  --cfg-options \
    env.max_completion_tokens=20480
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-1-flash/huoshan/huoshan \
    model.target=deepseek-v4-1-flash/huoshan/huoshan

python scripts/train.py \
  --config configs/spreadsheetbench/default.yaml \
  --cfg-options \
    env.max_completion_tokens=20480
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4-1-flash/huoshan/huoshan \
    model.target=deepseek-v4-1-flash/huoshan/huoshan

python scripts/train.py \
  --config configs/bird/default.yaml \
  --cfg-options \
    env.workers="${SKILLOPT_WORKERS}" \
    env.max_completion_tokens="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS}" \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=deepseek-v4.1-flash-0910/huoshan/huoshan \
    model.target=deepseek-v4.1-flash-0910/huoshan/huoshan
```

## 261003 self command - # self test
```bash
python test-phase/searchqa_test.py --mode cover --num-skills 5
python test-phase/searchqa_test.py --mode best_repeat --num-skills 5
python test-phase/searchqa_test.py --mode top_k --num-skills 5
python test-phase/searchqa_test.py --mode cover_vote_last --num-skills 5
python test-phase/searchqa_test.py --mode best_vote_global --num-skills 5
python test-phase/searchqa_test.py --mode vote_global --num-skills 5
python test-phase/searchqa_test.py --mode vote_global_milp --num-skills 5
python test-phase/searchqa_test.py --mode max_cover_milp --num-skills 5
python test-phase/searchqa_test.py --mode best_max_cover_milp --num-skills 5

export ALFWORLD_DATA="$HOME/.cache/alfworld"
python test-phase/alfworld_test.py \
  --input-root outputs/skillopt_alfworld_bailian-deepseek-v4-flash-0731_20261004_030059 \
  --mode best_repeat \
  --num-skills 5
  
python test-phase/spreadsheetbench_test.py \
  --input-root outputs/skillopt_spreadsheetbench_bailian-deepseek-v4-flash-0731_20261004_035613 \
  --mode top_k \
  --num-skills 5
python test-phase/spreadsheetbench_test.py \
  --input-root outputs/skillopt_spreadsheetbench_bailian-deepseek-v4-flash-0731_20261004_035613 \
  --mode top_k \
  --num-skills 5
  
python test-phase/officeqa_test.py \
  --input-root outputs/skillopt_officeqa_bailian-deepseek-v4-flash-0731_20261005_040355 \
  --mode top_k \
  --num-skills 5
  
python test-phase/livemathematicianbench_test.py \
  --input-root outputs/skillopt_livemathematicianbench_bailian-deepseek-v4-flash-0731_20261005_034349 \
  --mode top_k \
  --num-skills 5
python test-phase/livemathematicianbench_test.py \
  --input-root outputs/skillopt_livemathematicianbench_bailian-deepseek-v4-flash-0731_20261005_034349 \
  --mode best_repeat \
  --num-skills 5
```

## 261003 self command - self check
```bash
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_234948_vote_global_numskills5 \
  --dataset searchqa
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261004_015929_top_k_numskills9 \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315 \
  --dataset searchqa \
  --policy coalition \
  --override-margin 0.05
# re-generate questions with divergent answers with few-shot
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261006_021120_top_k_numskills3 \
  --dataset searchqa \
  --policy rag \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315
# ablation with rag: re-generate questions with divergent answers without few-shot
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261006_021120_top_k_numskills3 \
  --dataset searchqa \
  --policy no_rag \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315
# ablation with rag: three stage round by k=k
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261006_021120_top_k_numskills3 \
  --dataset searchqa \
  --policy rag_matrix \
  --matrix-samples 5

# ablation with fine-grain
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261004_015929_top_k_numskills9 \
  --dataset searchqa \
  --policy rag_plus \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261004_015929_top_k_numskills9 \
  --dataset searchqa \
  --policy rag_plus_history \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261004_015929_top_k_numskills9 \
  --dataset searchqa \
  --policy rag_plus_history_candidate \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261004_015929_top_k_numskills9 \
  --dataset searchqa \
  --policy rag_plus_history_show_self \
  --gate-root outputs/skillopt_searchqa_bailian-deepseek-v4-flash-0731_20261003_012315
  
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_officeqa_bailian-deepseek-v4-flash-0731_20261006_045154_top_k_numskills5 \
  --dataset officeqa \
  --policy rag \
  --gate-root outputs/skillopt_officeqa_bailian-deepseek-v4-flash-0731_20261005_040355
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_spreadsheetbench_bailian-deepseek-v4-flash-0731_20261006_045110_top_k_numskills5 \
  --dataset spreadsheetbench \
  --policy rag \
  --gate-root outputs/skillopt_spreadsheetbench_bailian-deepseek-v4-flash-0731_20261004_035613
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_livemathematicianbench_bailian-deepseek-v4-flash-0731_20261006_222705_top_k_numskills5 \
  --dataset livemathematicianbench \
  --policy rag \
  --gate-root outputs/skillopt_livemathematicianbench_bailian-deepseek-v4-flash-0731_20261005_034349

# 261012-1000 gpt-5.4-nano
python test-phase/test_vote_result.py \
  --input-root outputs/skillopt_searchqa_gpt-5.4-nano_20261008_062902_top_k_numskills5 \
  --dataset searchqa \
  --policy rag_plus_history_show_self \
  --gate-root outputs/skillopt_searchqa_gpt-5.4-nano_20261008_040547




# 261010-0752
append to command "--input-root outputs/skillopt_searchqa_gpt-5.4-nano_20261008_040547"
append to command "--input-root outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-huoshan_20261008_042358"
append to command "--input-root outputs/skillopt_searchqa_glm-5.3-flash-tencent-sfb_20261008_052127"
append to command "--input-root outputs/skillopt_searchqa_Qwen3.8-27B_20261008_050556"

python test-phase/searchqa_test.py --mode best_repeat --num-skills 5
python test-phase/searchqa_test.py --mode top_k --num-skills 5
python test-phase/searchqa_test.py --mode best_repeat --num-skills 7
python test-phase/searchqa_test.py --mode top_k --num-skills 7

# 261010-0752
append to command "--input-root outputs/skillopt_officeqa_gpt-5.4-nano_20261008_051639"
append to command "--input-root outputs/skillopt_officeqa_deepseek-v4-1-flash-huoshan-huoshan_20261008_153125"

python test-phase/officeqa_test.py --mode best_repeat --num-skills 5
python test-phase/officeqa_test.py --mode top_k --num-skills 5
python test-phase/officeqa_test.py --mode best_repeat --num-skills 7
python test-phase/officeqa_test.py --mode top_k --num-skills 7

# 261010-0752
append to command "--input-root outputs/skillopt_livemathematicianbench_gpt-5.4-nano_20261008_043258"
append to command "--input-root outputs/skillopt_livemathematicianbench_deepseek-v4-1-flash-huoshan-huoshan_20261008_102339"

python test-phase/livemathematicianbench_test.py --mode best_repeat --num-skills 5
python test-phase/livemathematicianbench_test.py --mode top_k --num-skills 5
python test-phase/livemathematicianbench_test.py --mode best_repeat --num-skills 7
python test-phase/livemathematicianbench_test.py --mode top_k --num-skills 7

# 261108 searchqa: deepseek-v4-1-flash + Qwen3.8-27B, top_k/best_repeat k=3,5,7 + rag_plus_history_show_self
# one tmux session per job; logs in logs/<session>.{infer,showself}.log
export OPENAI_COMPATIBLE_API_KEY="sk-xxx"
export EMBEDDING_API_KEY="sk-xxx"
bash test-phase/launch_searchqa_dpskv41_qwen.sh        # all 12 jobs
bash test-phase/launch_searchqa_dpskv41_qwen.sh dpsk   # only deepseek-v4-1-flash (6 jobs)
bash test-phase/launch_searchqa_dpskv41_qwen.sh qwen   # only Qwen3.8-27B (6 jobs)
SKILLOPT_WORKERS=12 bash test-phase/launch_searchqa_dpskv41_qwen.sh qwen   # lower per-job concurrency (default 24)
```
## 261120 early-stop experiments: SearchQA / TriviaQA / HotpotQA / BIRD

Reproduction notes for the validation-early-stop (`selection_early_stop`) and
training-early-stop (`train_early_stop`) experiments. Everything below is
recorded from the runs themselves (`config.json` + `summary.json` in
`outputs/<run>/`), not from memory. Runs live on `ssh cpu` under
`/mnt/sdd3/xuancheng/code/github/SkillOpt`.

### Context: two early stops

| switch | what it does | exact? |
|---|---|---|
| `evaluation.selection_early_stop` + `selection_early_stop_batch` | When validating one candidate, evaluate dev items in batches, ordered hardest-first by historical per-item score, and stop once `(correct_so_far + remaining) / n <= current_score`. The gate decision is identical to a full evaluation. Accepted candidates are always scored on every item. | **yes** — the reject decision is a bound argument |
| `evaluation.train_early_stop` (+ `_rho`, `_k`, `_delta`, `_lambda`, `_min_steps`) | End the whole run once recent candidates show no paired progress over the current skill: per step `z = (gained - lost) / sqrt(discordant)`, `cusum = max(0, cusum + z - delta)`, counter of steps with `cusum == 0`; stop after step `>= max(min_steps, ceil(rho * total_steps))` when the counter `>= k`. May stop mid-epoch, in which case the end-of-epoch slow update / meta skill are skipped. | **no** — a later step might still have improved the skill |

Both switches are off by default; with them off the training loop behaves exactly
as before. Implementation: `skillopt/evaluation/early_stop.py`,
`skillopt/evaluation/train_early_stop.py` (new in 7551cc5). Tests:
`tests/test_selection_early_stop.py`, `tests/test_train_early_stop.py`.

### Shared recipe

All 16 runs below use the same recipe; only the three items marked **[var]** change.
All 16 finished by 2026-11-21 01:49; the table holds the measured values.

```bash
# on ssh cpu, in /mnt/sdd3/xuancheng/code/github/SkillOpt
export OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS=131072
export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS=131072
export SKILLOPT_WORKERS=24                       # read by test-phase only
export OPENAI_COMPATIBLE_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-..."        # tokenhub key
export OPENAI_COMPATIBLE_MODEL="deepseek-v4.1-flash-1017/huoshan/sfc"

python scripts/train.py \
  --config configs/<bench>/default.yaml \
  --cfg-options \
    env.workers=24 \
    env.max_completion_tokens=131072 \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer="$OPENAI_COMPATIBLE_MODEL" \
    model.target="$OPENAI_COMPATIBLE_MODEL" \
    <early-stop flags> \
    env.out_root=outputs/<run name>
```

Effective config common to the runs listed here (values come from `configs/_base_/default.yaml`
unless overridden): `num_epochs=4`, `batch_size=40`, `accumulation=1`,
`seed=42`, `split_seed=42`, `edit_budget=4`, `lr_scheduler=cosine`,
`lr_control_mode=fixed`, `skill_update_mode=patch`, `use_gate=true`,
`gate_metric=hard`, `use_slow_update=true` with
`slow_update_gate_with_selection=false`, `use_meta_skill=true`,
`reasoning_effort=medium`, `max_completion_tokens=131072`.

**[var] model name.** Runs started 2026-11-18 and 2026-11-19 use
`deepseek-v4.1-flash-1017/huoshan/sfc`; runs started 2026-11-20 use
`deepseek-v4.1-flash-1017/huoshan_L/sfc`. The provider renamed the endpoint; the
served model is the same. The out_root names keep these apart:
`deepseek-v4.1-flash-1017` (plain) vs `deepseek-v4.1-flash-1017L` (`_L`).

**[var] early-stop flags.**

| arm | flags |
|---|---|
| FULL (baseline) | *(none)* |
| ES (dev early stop only) | `evaluation.selection_early_stop=true evaluation.selection_early_stop_batch=4` |
| AGG ρ=0.75 | the ES flags + `evaluation.train_early_stop=true evaluation.train_early_stop_rho=0.75` |
| AGG ρ=0.50 | the ES flags + `evaluation.train_early_stop=true evaluation.train_early_stop_rho=0.50` |

The other `train_early_stop_*` values stay at their defaults: `k=3`,
`delta=0.5`, `lambda=0.6`, `min_steps=10`.

**[var] benchmark config, splits, and run name.**

| bench | `--config` | `split_dir` | train / val / test | steps/epoch × 4 | `out_root` suffix |
|---|---|---|---|---|---|
| SearchQA | `configs/searchqa/default.yaml` | `data/searchqa_split` | 400 / 200 / 1,400 | 10 × 4 = 40 | `skillopt_searchqa_deepseek-v4.1-flash-1017[L]_<TS>_<arm>` |
| TriviaQA | `configs/triviaqa/default.yaml` | `data/triviaqa_split` | 1,600 / 200 / 1,000 | 40 × 4 = 160 | `skillopt_triviaqa_deepseek-v4.1-flash-1017[L]_<TS>_<arm>` |
| HotpotQA | `configs/hotpotqa/default.yaml` | `data/hotpotqa_split` | 1,600 / 200 / 1,000 | 40 × 4 = 160 | `skillopt_hotpotqa_deepseek-v4.1-flash-1017[L]_<TS>_<arm>` |
| BIRD | `configs/bird/default.yaml` | `data/bird_split` | 834 / 200 / 500 | 21 × 4 = 84 | `skillopt_bird_deepseek-v4.1-flash-1017[L]_<TS>_<arm>` |

`<arm>` is `earlystop_b4` (ES), `fullval` (FULL), `agg_b4_rho075`, or
`agg_b4_rho050`. `<TS>` is the timestamp in the run directory name.
BIRD additionally reads
`env.db_root=/mnt/sdd3/xuancheng/code/github/bird-text-to-sql-260820/data_dev/dev_databases`
(read-only) and uses `exec_timeout=300`, `sql_timeout=30`, both from its config.

### Data that must exist before reproducing

```bash
# SearchQA
python scripts/materialize_searchqa.py

# TriviaQA (rc.web) and HotpotQA (distractor), both via the SearchQA adapter
python scripts/materialize_triviaqa.py --output-dir data/triviaqa_split --train 1600 --val 200 --test 1000
python scripts/materialize_hotpotqa.py --output-dir data/hotpotqa_split --train 1600 --val 200 --test 1000

# BIRD (needs the prompt files from the bird-text-to-sql checkout)
python scripts/materialize_bird.py \
  --dev-jsonl /mnt/sdd3/xuancheng/code/github/bird-text-to-sql-260820/datas/dev_prompt.generated.jsonl \
  --mini-dev-jsonl /mnt/sdd3/xuancheng/code/github/bird-text-to-sql-260820/datas/mini_dev_prompt.generated.jsonl
```

Initial skills are the per-bench files listed as `env.skill_init` in each config
(`skillopt/envs/<bench>/skills/initial.md`).

### Run inventory and measured results

Everything is read from `outputs/<run>/summary.json`. Percentages are savings
against the matching FULL run of the same benchmark. `test best` is the best-on-dev
skill scored on the held-out test split; `test final` is the last skill.

| benchmark | run | model | steps run | stop step | best step | total tokens | train+dev tokens | test best | test final | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| SearchQA | ES (dev early stop, b=4) | 1017 | 40/40 | - | 4 | 32.45M (53.6%) | 20.11M (61.2%) | 0.8307 | 0.8386 | 3.7h |
| SearchQA | FULL (baseline) | 1017 | 40/40 | - | 38 | 69.88M | 51.82M | 0.8486 | 0.8536 | 2.7h |
| SearchQA | AGG rho=0.75 | 1017 | 30/40 | 30 | 7 | 30.85M (55.9%) | 17.25M (66.7%) | 0.8464 | 0.8450 | 2.6h |
| SearchQA | AGG rho=0.50 | 1017L | 20/40 | 20 | 14 | 19.28M (72.4%) | 11.98M (76.9%) | 0.8429 | 0.8429 | 1.8h |
| TriviaQA | ES (dev early stop, b=4) | 1017 | 160/160 | - | 97 | 99.14M (54.0%) | 88.08M (56.6%) | 0.8850 | 0.8970 | 8.0h |
| TriviaQA | FULL (baseline) | 1017 | 160/160 | - | 81 | 215.61M | 202.87M | 0.8910 | 0.8920 | 7.4h |
| TriviaQA | AGG rho=0.75 | 1017L | 120/160 | 120 | 62 | 90.52M (58.0%) | 78.36M (61.4%) | 0.8930 | 0.8960 | 5.4h |
| TriviaQA | AGG rho=0.50 | 1017L | 80/160 | 80 | 12 | 51.26M (76.2%) | 43.21M (78.7%) | 0.8830 | 0.8770 | 4.0h |
| HotpotQA | ES (dev early stop, b=4) | 1017 | 160/160 | - | 79 | 160.50M (28.6%) | 145.94M (31.6%) | 0.6780 | 0.6830 | 21.4h |
| HotpotQA | FULL (baseline) | 1017 | 160/160 | - | 5 | 224.63M | 213.35M | 0.6760 | 0.6780 | 14.6h |
| HotpotQA | AGG rho=0.75 | 1017L | 120/160 | 120 | 110 | 149.01M (33.7%) | 139.24M (34.7%) | 0.6860 | 0.6860 | 15.7h |
| HotpotQA | AGG rho=0.50 | 1017L | 80/160 | 80 | 70 | 72.27M (67.8%) | 65.86M (69.1%) | 0.6730 | 0.6730 | 11.5h |
| BIRD | ES (dev early stop, b=4) | 1017 | 84/84 | - | 63 | 136.43M (39.4%) | 120.89M (42.2%) | 0.7180 | 0.7380 | 23.6h |
| BIRD | FULL (baseline) | 1017 | 84/84 | - | 45 | 225.14M | 209.21M | 0.7100 | 0.7000 | 11.4h |
| BIRD | AGG rho=0.75 | 1017L | 63/84 | 63 | 37 | 93.73M (58.4%) | 80.99M (61.3%) | 0.7140 | 0.7140 | 16.8h |
| BIRD | AGG rho=0.50 | 1017L | 42/84 | 42 | 38 | 52.38M (76.7%) | 46.47M (77.8%) | 0.6820 | 0.6820 | 12.7h |

### rho comparison (aggregated from the table above)

All eight AGG runs stopped exactly at `ceil(rho * total_steps)` (30/120/80/63
for rho=0.75 and 20/80/42 for rho=0.50): in every case the data rule had already
fired well before the threshold, so rho alone decided the stop step.

Savings vs the matching FULL run, and test accuracy of the best-on-dev skill:

| bench | rho=0.75 total save | rho=0.75 train+dev save | rho=0.75 test best | rho=0.50 total save | rho=0.50 train+dev save | rho=0.50 test best |
|---|---|---|---|---|---|---|
| SearchQA | 55.9% | 66.7% | 0.8464 (FULL -0.21 pt) | 72.4% | 76.9% | 0.8429 (FULL -0.57 pt) |
| TriviaQA | 58.0% | 61.4% | 0.8930 (FULL +0.20 pt) | 76.2% | 78.7% | 0.8830 (FULL -0.80 pt) |
| HotpotQA | 33.7% | 34.7% | 0.6860 (FULL +1.00 pt) | 67.8% | 69.1% | 0.6730 (FULL -0.30 pt) |
| BIRD | 58.4% | 61.3% | 0.7140 (FULL +0.40 pt) | 76.7% | 77.8% | 0.6820 (FULL -2.80 pt) |

Paired per-item comparison of the rho=0.50 best skill against the rho=0.75 best
skill (discordant items only, from `test_eval/results.jsonl`):

| bench | items | rho=0.50 better | rho=0.75 better | net |
|---|---|---|---|---|
| SearchQA | 1,400 | 27 | 32 | -5 (-0.36 pt) |
| TriviaQA | 1,000 | 9 | 19 | -10 (-1.00 pt) |
| HotpotQA | 1,000 | 25 | 38 | -13 (-1.30 pt) |
| BIRD | 500 | 14 | 30 | -16 (-3.20 pt) |

Reading: rho=0.75 lands within +/-1 pt of FULL on all four benchmarks (one below,
three above). rho=0.50 buys another 30-52% of the remaining tokens and is within
noise on SearchQA (-0.57 pt) but loses 0.8-3.2 pt on the other three. All of these
are single runs; the starting-point noise measured on one test split was
0.29-1.00 pt, so the smaller gaps are not resolved.

Notes on the numbers:

- `total tokens` is `summary.json → token_summary._total.total_tokens`: every model
  call in the run, including the three test evaluations and the final dev
  re-evaluation. `train+dev tokens` sums the per-step `tokens` recorded in
  `history.json`, i.e. training steps only. The difference between them is the
  fixed evaluation cost (5.0%-25.8% of a run depending on the benchmark).
- Wall-clock differs from token savings because the two benches differ in
  per-item cost; HotpotQA's ES run took longer than its FULL run (21.4h vs 14.6h)
  because early stopping evaluates the *hardest* dev items first and those are
  the most expensive ones.
- The ES arm and the AGG arms are separate training runs with independent
  sampling, so their test scores differ by more than the early-stop flags alone
  would explain. Sampling noise was measured two ways: (a) three SearchQA runs
  (2026-11-11/12, `deepseek-v4.1-flash-0910`) from the identical initial skill
  scored 0.7671 / 0.7600 / 0.7571 on the same 1,400 items (0.29-1.00 pt of
  purely sampling-driven spread); (b) across the 16 runs here, the *test
  baseline of the unchanged initial skill* itself ranges 0.7421-0.7521
  (SearchQA), 0.8230-0.8330 (TriviaQA), 0.6030-0.6260 (HotpotQA) and
  0.6720-0.6940 (BIRD). Any table difference smaller than that range is not
  resolved by a single run.

### Per-run commands (exact `out_root` values)

Each run is the shared recipe with its **[var] early-stop flags**, its
**[var] benchmark config**, and this `out_root`. These are the actual `env.out_root` values, so re-running them
would overwrite the existing results — change the timestamp when re-running.

```bash
# SearchQA — ES / FULL (2026-11-18, deepseek-v4.1-flash-1017/huoshan/sfc)
env.out_root=outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-sfc_20261118_200115_earlystop_b4
env.out_root=outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-sfc_20261118_200115_fullval
# SearchQA — AGG rho=0.75 / rho=0.50
env.out_root=outputs/skillopt_searchqa_deepseek-v4.1-flash-1017_20261119_211735_agg_b4_rho075
env.out_root=outputs/skillopt_searchqa_deepseek-v4.1-flash-1017L_20261120_002105_agg_b4_rho050

# TriviaQA — ES / FULL (2026-11-18)
env.out_root=outputs/skillopt_triviaqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_earlystop_b4
env.out_root=outputs/skillopt_triviaqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_fullval
# TriviaQA — AGG rho=0.75 / rho=0.50
env.out_root=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
env.out_root=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050

# HotpotQA — ES / FULL (2026-11-18)
env.out_root=outputs/skillopt_hotpotqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_earlystop_b4
env.out_root=outputs/skillopt_hotpotqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_fullval
# HotpotQA — AGG rho=0.75 / rho=0.50
env.out_root=outputs/skillopt_hotpotqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
env.out_root=outputs/skillopt_hotpotqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050

# BIRD — ES / FULL (2026-11-18)
env.out_root=outputs/skillopt_bird_deepseek-v4.1-flash-1017_20261118_200644_earlystop_b4
env.out_root=outputs/skillopt_bird_deepseek-v4.1-flash-1017_20261118_073747_fullval
# BIRD — AGG rho=0.75 / rho=0.50
env.out_root=outputs/skillopt_bird_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
env.out_root=outputs/skillopt_bird_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050
```

### Where the evidence for each claim lives

| claim | file |
|---|---|
| which step each AGG run stopped at, and why | `outputs/<run>/summary.json → train_early_stop` (`stopped`, `stop_step`, `planned_steps`, `threshold_step`, `final_state`) |
| per-step `z`, cusum, flat counter | `outputs/<run>/history.json[*].train_early_stop` (`z`, `d`, `u`, `n`, `partial`, `state`) |
| how many dev items early stop actually evaluated | `outputs/<run>/history.json[*].selection_early_stop.n_evaluated` / `.n_total` |
| per-item dev results and the final test scores | `outputs/<run>/steps/step_*/selection_eval/results.jsonl`, `test_eval{,_baseline,_final}/results.jsonl` |
| per-item test results for best and final skills | `outputs/<run>/test_eval/results.jsonl`, `outputs/<run>/test_eval_final/results.jsonl` |

### Reproducing a result end to end

```bash
# 1. materialize the splits (see above), then
# 2. run the arm with the shared recipe and the arm's flags
# 3. check the early stop fired as expected
python - <<'PY'
import json
s = json.load(open("outputs/<run>/summary.json"))
print(s["train_early_stop"])
PY
# 4. compare with its FULL partner
python - <<'PY'
import json
for r in ("<full run>", "<agg run>"):
    s = json.load(open(f"outputs/{r}/summary.json"))
    print(r, s["token_summary"]["_total"]["total_tokens"], s["test_hard"], s["final_test_hard"])
PY
```

## TriviaQA、HotpotQA、BIRD：四个原版方法的 CPU 启动命令

2026-11-24 批次 `20261124-full-w12-r1` 使用固定划分和各方法的 FULL 默认训练流程。TriviaQA 和 HotpotQA 各为 1,600 / 200 / 1,000 题，BIRD 为 834 / 200 / 500 题。答题 worker 为 12，target 和 optimizer 的生成上限均为 131,072。SkillGen 额外使用 SiliconFlow 的 BAAI/bge-m3。四个仓库各自运行本仓库的程序。

从 Mac 执行 `ssh cpu`，在 CPU 的 Bash 中输入并导出两项凭据。已导出时可跳过这一步。

```bash
read -rsp 'TokenHub API key: ' OPENAI_COMPATIBLE_API_KEY; echo; export OPENAI_COMPATIBLE_API_KEY
read -rsp 'SiliconFlow embedding key: ' EMBEDDING_API_KEY; echo; export EMBEDDING_API_KEY
```

旧批次 `20261124-full-w12-r1` 的 TriviaQA/SkillGen 和 TriviaQA/SkillLens 已跑完，各有 1,000 条最终 test 结果。两个结果仍在旧批次的 `triviaqa/{skillgen,skilllens}/results/`。首次批次与 Lite 工作目录修复的启动记录分别见 Git 提交 `3ea1f36`、`d2886b8`。

下面是其余十组原版实验实际使用的并行命令。从 Mac 登录 `ssh cpu`、导出上述两项凭据后，在 CPU 的 Bash 中执行。每项直接启动一个独立 tmux 会话、各有 12 个答题 worker。新输出使用 `20261124-full-w12-parallel-r2`。重复实验时选用新的 `FULL12_BATCH` 和 `FULL12_TMUX_PREFIX`。

```bash
# CPU parallel launch (20261124-full-w12-parallel-r2)
(
  set -euo pipefail
  umask 077
  : "${OPENAI_COMPATIBLE_API_KEY:?export the TokenHub key first}"
  : "${EMBEDDING_API_KEY:?export the SiliconFlow key first}"
  export OPENAI_COMPATIBLE_API_KEY EMBEDDING_API_KEY
  base=/mnt/sdd3/xuancheng/code/github
  batch="$HOME/fixed-benchmark-runs/${FULL12_BATCH:-20261124-full-w12-parallel-r2}"
  prefix=${FULL12_TMUX_PREFIX:-full12p}
  py="$HOME/miniconda3/envs/rdagent/bin/python"
  evo_py="$HOME/miniconda3/bin/python"
  export OPENAI_COMPATIBLE_BASE_URL=https://tokenhub.sensetime.com/v1
  export OPENAI_COMPATIBLE_MODEL=deepseek-v4.1-flash-1017/huoshan_L/sfc
  export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS=131072 OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS=131072
  export SKILLOPT_WORKERS=12 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1/embeddings EMBEDDING_MODEL=BAAI/bge-m3
  export BIRD_DB_ROOT="$base/bird-text-to-sql-260820/data_dev/dev_databases"
  export PATH="$HOME/.opencode/bin:$PATH" OPENCODE_DISABLE_AUTOUPDATE=true OPENCODE_DISABLE_MODELS_FETCH=true
  unset TARGET_OPENAI_COMPATIBLE_MODEL OPTIMIZER_OPENAI_COMPATIBLE_MODEL SKILLOPT_LITE_TARGET_MODEL
  unset TARGET_OPENAI_COMPATIBLE_BASE_URL OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL
  unset TARGET_OPENAI_COMPATIBLE_API_KEY OPTIMIZER_OPENAI_COMPATIBLE_API_KEY

  declare -A repos=([skillgen]=SkillGen [skilllens]=SkillLens [skillopt-lite]=SkillOpt-Lite [evoskill]=EvoSkill)
  declare -A revs=([skillgen]=68f04e3 [skilllens]=8396f7a [skillopt-lite]=ffd1914 [evoskill]=32a8640)
  jobs=(
    'triviaqa skillopt-lite' 'triviaqa evoskill'
    'hotpotqa skillgen' 'hotpotqa skillopt-lite' 'hotpotqa skilllens' 'hotpotqa evoskill'
    'bird skillgen' 'bird skillopt-lite' 'bird skilllens' 'bird evoskill'
  )
  command -v tmux >/dev/null
  command -v opencode >/dev/null
  test -x "$py"
  test -x "$evo_py"
  for method in "${!repos[@]}"; do
    test "$(git -C "$base/${repos[$method]}" rev-parse --short=7 HEAD)" = "${revs[$method]}"
  done
  for spec in "${jobs[@]}"; do
    read -r bench method <<< "$spec"
    if tmux -L "$prefix-$bench-$method" list-sessions >/dev/null 2>&1; then
      echo "Existing tmux session: $prefix-$bench-$method" >&2
      exit 1
    fi
  done
  mkdir -p "$(dirname "$batch")"
  mkdir "$batch"

  launch() (
    set -euo pipefail
    bench=$1; method=$2
    repo="$base/${repos[$method]}"
    session="$prefix-$bench-$method"
    split="$base/SkillOpt/data/${bench}_split"
    export FULL12_OUT="$batch/$bench/$method" FULL12_REV="${revs[$method]}"
    mkdir -p "$batch/$bench"
    mkdir "$FULL12_OUT"
    if [[ "$method" != skillgen ]]; then unset EMBEDDING_API_KEY; fi
    case "$method" in
      skillgen)
        cmd=("$py" -B -u -m benchmarks.searchqa --benchmark "$bench" --config "$repo/config.yaml"
             --split-dir "$split" --output "$FULL12_OUT/results")
        ;;
      skilllens)
        cmd=("$py" -B -u -m skilllens.searchqa --benchmark "$bench" --split-dir "$split"
             --extract-config "$repo/configs/examples/seal0_parallel.yaml" --provider openai --api-style chat
             --workers "$SKILLOPT_WORKERS" --target-model "$OPENAI_COMPATIBLE_MODEL"
             --extract-model "$OPENAI_COMPATIBLE_MODEL" --output "$FULL12_OUT/results")
        ;;
      skillopt-lite)
        export SKILLOPT_AUTH_MODE=openai_compatible SKILLOPT_LITE_PYTHON="$py"
        export XDG_DATA_HOME="$FULL12_OUT/opencode-data" XDG_STATE_HOME="$FULL12_OUT/opencode-state"
        export SKILLOPT_DONE_FILE="$FULL12_OUT/phase-notifications.log"
        export "${bench^^}_SPLIT_DIR=$split"
        cmd=("$py" -B -u copilot_example/searchqa/opencode_loop.py --benchmark "$bench"
             --rounds 3 --batch 20 --run-dir "$FULL12_OUT/host")
        ;;
      evoskill)
        export PYTHONPATH="$HOME/.cache/uv/archive-v0/Xgr_JH5lu1epaTUF"
        GIT_AUTHOR_NAME=$(git -C "$repo" config --get user.name)
        GIT_AUTHOR_EMAIL=$(git -C "$repo" config --get user.email)
        : "${GIT_AUTHOR_NAME:?missing Git author name}" "${GIT_AUTHOR_EMAIL:?missing Git author email}"
        export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL
        export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
        cmd=("$evo_py" -B -u examples/searchqa/run.py --benchmark "$bench" --harness openai-compatible
             --model "$OPENAI_COMPATIBLE_MODEL" --workers "$SKILLOPT_WORKERS"
             --split-dir "$split" --work-dir "$FULL12_OUT/workspace")
        ;;
    esac
    tmux -L "$session" -f /dev/null new-session -d -s "$session" -c "$repo" bash -c '
      set -euo pipefail
      umask 077
      exec > >(tee -a "$FULL12_OUT/console.log") 2>&1
      finish() { rc=$?; printf "%s\n" "$rc" > "$FULL12_OUT/exit.code"; date -Is > "$FULL12_OUT/finished.at"; }
      trap finish EXIT
      test "$(git rev-parse --short=7 HEAD)" = "$FULL12_REV"
      date -Is | tee "$FULL12_OUT/started.at"
      printf "repo=%s commit=%s model=%s workers=%s\n" "$PWD" "$(git rev-parse HEAD)" "$OPENAI_COMPATIBLE_MODEL" "$SKILLOPT_WORKERS"
      printf "argv:"; printf " %q" "$@"; printf "\n"
      "$@"
    ' _ "${cmd[@]}"
    echo "Launched $session"
  )
  for spec in "${jobs[@]}"; do
    read -r bench method <<< "$spec"
    launch "$bench" "$method"
  done
)
```

新批次位于 `$HOME/fixed-benchmark-runs/20261124-full-w12-parallel-r2/<bench>/<method>/`。每组的启动时间、实际参数和输出保存在 `started.at` 与 `console.log` 中，结束后查看 `exit.code` 及对应 test 汇总。查看某组可运行 `tmux -L full12p-bird-evoskill attach -t full12p-bird-evoskill`。

| 方法 | 本批每组目录内的结果位置 |
|---|---|
| SkillGen | `results/summary.json`、`results/baseline.jsonl`，部署候选后另有 `results/skill.jsonl` |
| SkillOpt-Lite | `host/workspace/.skillopt/_eval_run/test_*/eval_summary.json` 和同目录 `results.jsonl` |
| SkillLens | `results/test/eval_summary.json` 和 `results/test/results.jsonl` |
| EvoSkill | `workspace/.evoskill/test_summary.json` 和 `workspace/.evoskill/test_results.jsonl` |

## CPU 实验结果备份清单

以下绝对路径位于 `ssh cpu` 服务器，最近核对日期为 2026-11-24。备份整目录时请包含隐藏文件。SkillOpt 每个运行目录都包含 `summary.json`、`history.json` 和三份逐题 test 结果：`test_eval_baseline/`（训练前）、`test_eval/`（dev 选出的最佳 skill）、`test_eval_final/`（最后一步）。

**SkillOpt-Lite 的日志目录和实际工作区都要备份。SkillGen 重跑的最终备份以 `exit.status` 为 0 且 `test/summary.json` 已写入为准。**

| 序号 | 方法 / Benchmark | 配置或用途 | CPU 上的绝对路径 |
|---:|---|---|---|
| 1 | SkillOpt / SearchQA | 完整训练 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-sfc_20261118_200115_fullval/` |
| 2 | SkillOpt / SearchQA | 温和早停 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-sfc_20261118_200115_earlystop_b4/` |
| 3 | SkillOpt / SearchQA | 温和＋激进早停，ρ=0.75 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_searchqa_deepseek-v4.1-flash-1017_20261119_211735_agg_b4_rho075/` |
| 4 | SkillOpt / SearchQA | 温和＋激进早停，ρ=0.50 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_searchqa_deepseek-v4.1-flash-1017L_20261120_002105_agg_b4_rho050/` |
| 5 | SkillOpt / TriviaQA | 完整训练 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_fullval/` |
| 6 | SkillOpt / TriviaQA | 温和早停 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_earlystop_b4/` |
| 7 | SkillOpt / TriviaQA | 温和＋激进早停，ρ=0.75 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075/` |
| 8 | SkillOpt / TriviaQA | 温和＋激进早停，ρ=0.50 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050/` |
| 9 | SkillOpt / HotpotQA | 完整训练 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_hotpotqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_fullval/` |
| 10 | SkillOpt / HotpotQA | 温和早停 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_hotpotqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_earlystop_b4/` |
| 11 | SkillOpt / HotpotQA | 温和＋激进早停，ρ=0.75 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_hotpotqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075/` |
| 12 | SkillOpt / HotpotQA | 温和＋激进早停，ρ=0.50 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_hotpotqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050/` |
| 13 | SkillOpt / BIRD | 完整训练 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_bird_deepseek-v4.1-flash-1017_20261118_073747_fullval/` |
| 14 | SkillOpt / BIRD | 温和早停 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_bird_deepseek-v4.1-flash-1017_20261118_200644_earlystop_b4/` |
| 15 | SkillOpt / BIRD | 温和＋激进早停，ρ=0.75 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_bird_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075/` |
| 16 | SkillOpt / BIRD | 温和＋激进早停，ρ=0.50 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_bird_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050/` |
| 17 | SkillGen / SearchQA | 从头重跑。训练轨迹、候选 skill 和日志在此，训练前后 test 对照汇总写入 `test/summary.json` | `/mnt/sdd3/xuancheng/searchqa-original-runs/20261124-r2/skillgen/` |
| 18 | SkillGen / SearchQA | 首次运行因限流中断，保留了初始训练轨迹、日志和退出码 | `/mnt/sdd3/xuancheng/searchqa-original-runs/20261124-r1/skillgen/` |
| 19 | SkillLens / SearchQA | 已完成。含 `results/train/`、`results/val/`、`results/test/`、提炼记录和最终 `results/skill.md` | `/mnt/sdd3/xuancheng/searchqa-original-runs/20261124-r1/skilllens/` |
| 20 | SkillOpt-Lite / SearchQA | 已完成。启动日志、退出码和指向下一行的 `workspace` 符号链接 | `/mnt/sdd3/xuancheng/searchqa-original-runs/20261124-r1/skillopt-lite/` |
| 21 | SkillOpt-Lite / SearchQA | 实际工作区。含 `skill.md`、`.skillopt/history/` 和逐题结果，最终 test 在 `.skillopt/_eval_run/test_lgalkkoj/` | `/mnt/sdd3/xuancheng/code/github/SkillOpt-Lite/copilot_example/searchqa/workspaces/deepseek-v4.1-flash-1017-huoshan_L-sfc/` |
| 22 | EvoSkill / SearchQA | 已完成的第二次运行。保留隐藏目录，包括 `workspace/.git/` 的候选版本、`workspace/.claude/skills/` 的 skill 和 `workspace/.evoskill/` 的训练与 test 结果 | `/mnt/sdd3/xuancheng/searchqa-original-runs/20261124-r1/evoskill-attempt2/` |
| 23 | EvoSkill / SearchQA | 首次运行在初始化时退出，含日志和退出码 | `/mnt/sdd3/xuancheng/searchqa-original-runs/20261124-r1/evoskill/` |
| 24 | 输入 / SearchQA | 五个方法共用的 400 道 train、200 道 dev、1,400 道 test 及固定题目清单 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/data/searchqa_split/` |
| 25 | 输入 / TriviaQA | 固定 train、dev、test 题及清单 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/data/triviaqa_split/` |
| 26 | 输入 / HotpotQA | 固定 train、dev、test 题及清单 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/data/hotpotqa_split/` |
| 27 | 输入 / BIRD | 固定 train、dev、test 题及清单 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/data/bird_split/` |
| 28 | 输入 / BIRD | 执行和评测 SQL 用的数据库，与 BIRD 题目一同备份 | `/mnt/sdd3/xuancheng/code/github/bird-text-to-sql-260820/data_dev/dev_databases/` |
| 29 | SkillOpt-Lite / OpenCode | 可选。需要原始优化对话和调用用量时，从共享库导出本次会话 `ses_df5a18fe6ffeEKY7k4Nqr5oaPW` | `/mnt/sdd3/xuancheng/.local/share/opencode/opencode.db` |
