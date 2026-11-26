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

默认的完整实验（`FULL12_MODE=all`）是 4 个方法各跑 TriviaQA、HotpotQA、BIRD，共 12 组。每组使用独立 tmux 并行运行，答题并发为每组 12。默认批次和各任务的结果目录名均带模型部署名（斜杠替换为连字符）和北京时间批次启动时间戳。历史补跑模式 `ten` 跳过当时已经完成的两组 TriviaQA，`lite-retry` 只重试三个失败的 Lite 任务。重复运行时设置新的 `FULL12_BATCH` 和 `FULL12_TMUX_PREFIX`。

```bash
# CPU parallel launch (4 methods x 3 benchmarks)
(
  set -euo pipefail
  umask 077
  : "${OPENAI_COMPATIBLE_API_KEY:?export the TokenHub key first}"
  export OPENAI_COMPATIBLE_API_KEY
  mode=${FULL12_MODE:-all}
  case "$mode" in
    all|ten) : "${EMBEDDING_API_KEY:?export the SiliconFlow key first}"; export EMBEDDING_API_KEY ;;
    lite-retry) ;;
    *) echo "Invalid FULL12_MODE: $mode" >&2; exit 1 ;;
  esac
  base=/mnt/sdd3/xuancheng/code/github
  export OPENAI_COMPATIBLE_MODEL=deepseek-v4.1-flash-1017/huoshan_L/sfc
  default_batch=20261124-full-w12-parallel-r2
  default_prefix=full12p
  if [[ "$mode" == all ]]; then
    stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
    model_tag=${OPENAI_COMPATIBLE_MODEL//\//-}
    default_batch="full-w12_${model_tag}_${stamp}"
    default_prefix="full12all-$stamp"
  fi
  batch_name=${FULL12_BATCH:-$default_batch}
  legacy_batch="$HOME/fixed-benchmark-runs/$batch_name"
  declare -A bench_dirs
  for bench in triviaqa hotpotqa bird; do
    if [[ "$mode" == all ]]; then
      bench_dirs[$bench]="$HOME/${bench}-original-runs/$batch_name"
    else
      bench_dirs[$bench]="$legacy_batch/$bench"
    fi
  done
  prefix=${FULL12_TMUX_PREFIX:-$default_prefix}
  py="$HOME/miniconda3/envs/rdagent/bin/python"
  evo_py="$HOME/miniconda3/bin/python"
  export OPENAI_COMPATIBLE_BASE_URL=https://tokenhub.sensetime.com/v1
  export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS=131072 OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS=131072
  export SKILLOPT_WORKERS=12 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1/embeddings EMBEDDING_MODEL=BAAI/bge-m3
  export BIRD_DB_ROOT="$base/bird-text-to-sql-260820/data_dev/dev_databases"
  export PATH="$HOME/.opencode/bin:$PATH" OPENCODE_DISABLE_AUTOUPDATE=true OPENCODE_DISABLE_MODELS_FETCH=true
  unset TARGET_OPENAI_COMPATIBLE_MODEL OPTIMIZER_OPENAI_COMPATIBLE_MODEL SKILLOPT_LITE_TARGET_MODEL
  unset TARGET_OPENAI_COMPATIBLE_BASE_URL OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL
  unset TARGET_OPENAI_COMPATIBLE_API_KEY OPTIMIZER_OPENAI_COMPATIBLE_API_KEY

  declare -A repos=([skillgen]=SkillGen [skilllens]=SkillLens [skillopt-lite]=SkillOpt-Lite [evoskill]=EvoSkill)
  declare -A revs=([skillgen]=68f04e3 [skilllens]=8396f7a [skillopt-lite]=8c37ea3 [evoskill]=32a8640)
  if [[ "$mode" == all ]]; then revs[skillopt-lite]=fc34dd5; fi
  jobs=()
  for bench in triviaqa hotpotqa bird; do
    for method in skillgen skillopt-lite skilllens evoskill; do
      if [[ "$mode" == ten && "$bench" == triviaqa && ( "$method" == skillgen || "$method" == skilllens ) ]]; then continue; fi
      if [[ "$mode" == lite-retry && "$method" != skillopt-lite ]]; then continue; fi
      jobs+=("$bench $method")
    done
  done
  suffix=
  if [[ "$mode" == lite-retry ]]; then
    suffix=-attempt2
    test -d "$legacy_batch"
  fi
  directory_suffix=$suffix
  if [[ "$mode" == all ]]; then directory_suffix="_${model_tag}_${stamp}"; fi
  command -v tmux >/dev/null
  command -v opencode >/dev/null
  test -x "$py"
  test -x "$evo_py"
  for method in "${!repos[@]}"; do
    test "$(git -C "$base/${repos[$method]}" rev-parse --short=7 HEAD)" = "${revs[$method]}"
  done
  for spec in "${jobs[@]}"; do
    read -r bench method <<< "$spec"
    if tmux -L "$prefix-$bench-$method$suffix" list-sessions >/dev/null 2>&1; then
      echo "Existing tmux session: $prefix-$bench-$method$suffix" >&2
      exit 1
    fi
    test ! -e "${bench_dirs[$bench]}/$method$directory_suffix"
    if [[ "$mode" == lite-retry ]]; then
      test -s "${bench_dirs[$bench]}/$method/exit.code"
      test "$(cat "${bench_dirs[$bench]}/$method/exit.code")" != 0
      if tmux -L "$prefix-$bench-$method" has-session -t "$prefix-$bench-$method" 2>/dev/null; then
        echo "First attempt still running: $bench/$method" >&2; exit 1
      fi
    fi
  done
  if [[ "$mode" == all ]]; then
    for dir in "${bench_dirs[@]}"; do test ! -e "$dir"; done
    for dir in "${bench_dirs[@]}"; do mkdir -p "$(dirname "$dir")"; mkdir "$dir"; done
  elif [[ "$mode" == ten ]]; then
    mkdir -p "$(dirname "$legacy_batch")"
    mkdir "$legacy_batch"
  fi

  launch() (
    set -euo pipefail
    bench=$1; method=$2
    repo="$base/${repos[$method]}"
    session="$prefix-$bench-$method$suffix"
    split="$base/SkillOpt/data/${bench}_split"
    export FULL12_OUT="${bench_dirs[$bench]}/$method$directory_suffix" FULL12_REV="${revs[$method]}"
    mkdir -p "${bench_dirs[$bench]}"
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
    printf "Launched %s -> %s\n" "$session" "$FULL12_OUT"
  )
  for spec in "${jobs[@]}"; do
    read -r bench method <<< "$spec"
    launch "$bench" "$method"
  done
)
```

新完整批次按 benchmark 归档，与 `~/searchqa-original-runs/` 并列。三个根目录为 `~/triviaqa-original-runs/`、`~/hotpotqa-original-runs/` 和 `~/bird-original-runs/`。每个根目录内使用 `full-w12_<模型>_<YYYYMMDD_HHMMSS>/<method>_<模型>_<YYYYMMDD_HHMMSS>/`，12 组共用一次生成的北京时间批次时间戳。每组实际启动时间见 `started.at`，完整模型名与参数见 `console.log`。结束后查看 `exit.code` 和 test 汇总。连接会话时使用启动输出中的名称：`tmux -L "$session" attach -t "$session"`。

新 FULL 使用的 Lite 提交为 `fc34dd5`。其隔离宿主预建 `workspace/.skillopt/phase-logs/`，并给优化器提供在此写入阶段日志、退出标记的命令。

上一批（2026-11-25 00:26:51，北京时间）使用提交 `93a2d24` 的命令，模型为 `deepseek-v4.1-flash-1017/huoshan_L/sfc`。该批现有路径为 `/mnt/sdd3/xuancheng/fixed-benchmark-runs/full-w12-all-r1/<bench>/<method>/`，会话前缀为 `full12all`。

并行批次首次调度使用提交 `d34d2ee` 的启动命令。三组 Lite 首次尝试在首轮 dev 前退出，原始日志保存在 `<bench>/skillopt-lite/`。补跑时在原批次的 CPU shell 设置 `export FULL12_MODE=lite-retry`，仅提供 TokenHub key，然后执行上面完整的启动块。三组修复版 Lite 立即并行，使用 `<bench>/skillopt-lite-attempt2/` 和同名后缀的 tmux；修复版 `8c37ea3` 在各自 host 中准备初始 skill 和只含 Git 跟踪源码的阶段运行文件，先执行原版初始 dev、最佳快照和 seed=1 的首批 train，然后由 OpenCode 接续三轮优化。成功以完整逐题 test、汇总文件和退出码共同确认。复现十组全新运行时使用 `FULL12_MODE=ten`、新的批次和会话前缀。

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

### 首次有效运行结果目录（批次 `20261125_004054`）

下表记录四个 baseline 在 TriviaQA、HotpotQA 和 BIRD 上的 12 组完整训练结果。已在 CPU 服务器逐项核实目录存在、退出码为 `0`，且最终 test 汇总分别覆盖 1,000、1,000 和 500 题。

其余 20 组首次有效结果见上方备份清单：第 1–16 项为 SkillOpt 的四种配置，第 17、19、20–21、22 项为四个 baseline 的 SearchQA 结果，其中第 20–21 项对应同一次 SkillOpt-Lite 运行。

| Benchmark | 方法 | CPU 上的绝对路径 |
|---|---|---|
| TriviaQA | SkillGen | `/mnt/sdd3/xuancheng/triviaqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| TriviaQA | SkillOpt-Lite | `/mnt/sdd3/xuancheng/triviaqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| TriviaQA | SkillLens | `/mnt/sdd3/xuancheng/triviaqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| TriviaQA | EvoSkill | `/mnt/sdd3/xuancheng/triviaqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| HotpotQA | SkillGen | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| HotpotQA | SkillOpt-Lite | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| HotpotQA | SkillLens | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| HotpotQA | EvoSkill | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| BIRD | SkillGen | `/mnt/sdd3/xuancheng/bird-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| BIRD | SkillOpt-Lite | `/mnt/sdd3/xuancheng/bird-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| BIRD | SkillLens | `/mnt/sdd3/xuancheng/bird-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |
| BIRD | EvoSkill | `/mnt/sdd3/xuancheng/bird-original-runs/full-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_004054/` |

### 四个 baseline 的第 2、3 次重复运行（批次 `20261125_043046`）

本批次使用 `deepseek-v4.1-flash-1017/huoshan_L/sfc`，每组答题并发 12。下表列出全部 32 组已完成运行在 CPU 上的绝对路径。SearchQA 第 2 次、BIRD 第 2 和第 3 次的 SkillOpt-Lite 补跑目录使用时间戳 `20261125_045615`。

| Benchmark | 方法 | 第几次运行 | CPU 上的绝对路径 |
|---|---|---:|---|
| SearchQA | SkillGen | 2 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| SearchQA | SkillGen | 3 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| SearchQA | SkillOpt-Lite | 2 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_045615/` |
| SearchQA | SkillOpt-Lite | 3 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite-attempt3_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043318/` |
| SearchQA | SkillLens | 2 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| SearchQA | SkillLens | 3 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| SearchQA | EvoSkill | 2 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| SearchQA | EvoSkill | 3 | `/mnt/sdd3/xuancheng/searchqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | SkillGen | 2 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | SkillGen | 3 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | SkillOpt-Lite | 2 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | SkillOpt-Lite | 3 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite-attempt2_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043408/` |
| TriviaQA | SkillLens | 2 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | SkillLens | 3 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | EvoSkill | 2 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| TriviaQA | EvoSkill | 3 | `/mnt/sdd3/xuancheng/triviaqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | SkillGen | 2 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | SkillGen | 3 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | SkillOpt-Lite | 2 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | SkillOpt-Lite | 3 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite-retry_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043502/` |
| HotpotQA | SkillLens | 2 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | SkillLens | 3 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | EvoSkill | 2 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| HotpotQA | EvoSkill | 3 | `/mnt/sdd3/xuancheng/hotpotqa-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| BIRD | SkillGen | 2 | `/mnt/sdd3/xuancheng/bird-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| BIRD | SkillGen | 3 | `/mnt/sdd3/xuancheng/bird-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillgen_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| BIRD | SkillOpt-Lite | 2 | `/mnt/sdd3/xuancheng/bird-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_045615/` |
| BIRD | SkillOpt-Lite | 3 | `/mnt/sdd3/xuancheng/bird-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skillopt-lite_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_045615/` |
| BIRD | SkillLens | 2 | `/mnt/sdd3/xuancheng/bird-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| BIRD | SkillLens | 3 | `/mnt/sdd3/xuancheng/bird-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/skilllens_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| BIRD | EvoSkill | 2 | `/mnt/sdd3/xuancheng/bird-original-runs/rep2-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |
| BIRD | EvoSkill | 3 | `/mnt/sdd3/xuancheng/bird-original-runs/rep3-w12_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/evoskill_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261125_043046/` |

## 四个 baseline 在四个 benchmark 上的三次完整运行

成本为训练＋dev 的 LLM tokens，排除 test 和 embedding。精度含所有固定 test 题。SD 为三次运行的样本标准差（n−1）。两列提升均以百分点表示，用未舍入的精度计算。SkillOpt 使用唯一一次完整训练的前测作固定参考，SkillGen 先按运行编号减去同次前测，再取三次差值的平均。SearchQA 首次运行的答题并发为 24，后两次为 12，其余三个 benchmark 为每组 12。

### SearchQA（1,400 题，EM）

| 方法 | 第 1 次 tokens / 精度 | 第 2 次 tokens / 精度 | 第 3 次 tokens / 精度 | tokens 均值 ± SD | 精度均值 ± SD（%） | 较 SkillOpt 前测 | 较 SkillGen 前测 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SkillOpt（训练前） | 0 / 74.79% | — | — | 0 | 74.79（1 次） | 0.00 | +7.86 |
| SkillGen（训练前） | 0 / 65.36% | 0 / 67.93% | 0 / 67.50% | 0 | 66.93 ± 1.38 | −7.86 | 0.00 |
| SkillGen | 2,518,901 / 78.86% | 2,503,545 / 80.21% | 2,523,676 / 79.86% | 2,515,374 ± 10,519 | 79.64 ± 0.70 | +4.86 | +12.71 |
| SkillOpt-Lite | 2,148,941 / 80.79% | 2,987,216 / 81.71% | 2,273,981 / 80.93% | 2,470,046 ± 452,225 | **81.14 ± 0.50** | +6.36 | +14.21 |
| SkillLens | 2,472,492 / 79.86% | 2,335,085 / 79.71% | 2,335,787 / 80.71% | **2,381,121 ± 79,130** | 80.10 ± 0.54 | +5.31 | +13.17 |
| EvoSkill | 7,010,456 / 82.14% | 6,220,302 / 75.93% | 6,662,135 / 80.79% | 6,630,964 ± 395,998 | 79.62 ± 3.27 | +4.83 | +12.69 |

### TriviaQA（1,000 题，EM）

| 方法 | 第 1 次 tokens / 精度 | 第 2 次 tokens / 精度 | 第 3 次 tokens / 精度 | tokens 均值 ± SD | 精度均值 ± SD（%） | 较 SkillOpt 前测 | 较 SkillGen 前测 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SkillOpt（训练前） | 0 / 82.40% | — | — | 0 | 82.40（1 次） | 0.00 | +10.67 |
| SkillGen（训练前） | 0 / 72.00% | 0 / 70.50% | 0 / 72.70% | 0 | 71.73 ± 1.12 | −10.67 | 0.00 |
| SkillGen | 6,150,489 / 85.40% | 6,190,215 / 81.00% | 6,182,474 / 86.20% | 6,174,393 ± 21,060 | 84.20 ± 2.80 | +1.80 | +12.47 |
| SkillOpt-Lite | 2,323,971 / 89.50% | 2,426,014 / 88.80% | 2,228,699 / 89.80% | **2,326,228 ± 98,677** | **89.37 ± 0.51** | +6.97 | +17.63 |
| SkillLens | 7,689,963 / 84.40% | 7,725,557 / 85.20% | 7,897,892 / 86.00% | 7,771,137 ± 111,206 | 85.20 ± 0.80 | +2.80 | +13.47 |
| EvoSkill | 15,600,116 / 88.50% | 15,667,077 / 88.40% | 19,915,305 / 89.10% | 17,060,833 ± 2,472,272 | 88.67 ± 0.38 | +6.27 | +16.93 |

### HotpotQA（1,000 题，EM）

| 方法 | 第 1 次 tokens / 精度 | 第 2 次 tokens / 精度 | 第 3 次 tokens / 精度 | tokens 均值 ± SD | 精度均值 ± SD（%） | 较 SkillOpt 前测 | 较 SkillGen 前测 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SkillOpt（训练前） | 0 / 61.50% | — | — | 0 | 61.50（1 次） | 0.00 | +14.13 |
| SkillGen（训练前） | 0 / 47.70% | 0 / 47.00% | 0 / 47.40% | 0 | 47.37 ± 0.35 | −14.13 | 0.00 |
| SkillGen | 8,080,727 / 63.90% | 7,965,777 / 65.00% | 7,949,889 / 65.70% | 7,998,798 ± 71,396 | 64.87 ± 0.91 | +3.37 | +17.50 |
| SkillOpt-Lite | 2,672,117 / 68.20% | 2,766,336 / 67.00% | 2,488,183 / 67.00% | **2,642,212 ± 141,467** | **67.40 ± 0.69** | +5.90 | +20.03 |
| SkillLens | 9,970,401 / 64.50% | 10,053,901 / 62.70% | 10,187,612 / 62.70% | 10,070,638 ± 109,568 | 63.30 ± 1.04 | +1.80 | +15.93 |
| EvoSkill | 12,915,802 / 66.40% | 12,169,812 / 66.40% | 14,382,661 / 68.70% | 13,156,092 ± 1,125,824 | 67.17 ± 1.33 | +5.67 | +19.80 |

### BIRD（500 题，EX）

| 方法 | 第 1 次 tokens / 精度 | 第 2 次 tokens / 精度 | 第 3 次 tokens / 精度 | tokens 均值 ± SD | 精度均值 ± SD（%） | 较 SkillOpt 前测 | 较 SkillGen 前测 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SkillOpt（训练前） | 0 / 69.00% | — | — | 0 | 69.00（1 次） | 0.00 | +1.60 |
| SkillGen（训练前） | 0 / 67.00% | 0 / 68.40% | 0 / 66.80% | 0 | 67.40 ± 0.87 | −1.60 | 0.00 |
| SkillGen | 17,822,881 / 67.20% | 14,017,967 / 70.00% | 14,310,759 / 67.80% | 15,383,869 ± 2,117,313 | **68.33 ± 1.47** | −0.67 | +0.93 |
| SkillOpt-Lite | 5,245,074 / 67.80% | 4,535,358 / 69.20% | 4,663,199 / 67.80% | **4,814,544 ± 378,290** | 68.27 ± 0.81 | −0.73 | +0.87 |
| SkillLens | 8,809,111 / 67.20% | 8,768,319 / 68.40% | 8,838,017 / 68.40% | 8,805,149 ± 35,018 | 68.00 ± 0.69 | −1.00 | +0.60 |
| EvoSkill | 9,118,459 / 65.20% | 10,590,778 / 65.20% | 14,141,648 / 68.80% | 11,283,628 ± 2,582,274 | 66.40 ± 2.08 | −2.60 | −1.00 |

## SkillOpt 第 2 轮复现命令（CPU 批次 `20261126_080455`）

2026-11-26 在 CPU 上以提交 `595285dde10ece3842a342179d6449de337b175c` 启动了 16 个独立 tmux。SearchQA、TriviaQA、HotpotQA 和 BIRD 各跑四组：完整训练、温和早停、温和＋激进 ρ=0.75、温和＋激进 ρ=0.50。每组从对应 benchmark 的初始 skill 和 CPU 上已有的固定 split 开始，使用同一模型做优化和答题。每组答题并发 24，温和早停的 dev 批大小为 4。两个角色的 `131072` 均为输出 token 上限，实际请求还受调用处的输出预算约束。

在本地运行 `ssh cpu`，然后在 CPU 的 Bash 中执行下面的命令块。代码使用本批次的 Python 环境、配置文件和训练参数；再次执行时自动生成新的北京时间戳，在首轮结果的同级目录创建新结果。清单最后一列是对应首轮目录名，用于在日志中记录来源。

```bash
(
  set -euo pipefail
  umask 077
  repo=/mnt/sdd3/xuancheng/code/github/SkillOpt
  cd "$repo"
  reference_rev=595285dde10ece3842a342179d6449de337b175c
  git diff --quiet "$reference_rev" -- scripts skillopt configs
  revision=$(git rev-parse HEAD)
  py="$repo/.venv/bin/python"
  test -x "$py"
  command -v tmux >/dev/null

  if [[ -z ${OPENAI_COMPATIBLE_API_KEY:-} ]]; then
    read -rsp 'TokenHub API key: ' OPENAI_COMPATIBLE_API_KEY
    echo
  fi
  : "${OPENAI_COMPATIBLE_API_KEY:?TokenHub API key required}"
  export OPENAI_COMPATIBLE_API_KEY
  # 与原批次相同，清除按角色设置的旧覆盖值，再设置两个角色的输出上限。
  for key in ${!OPTIMIZER_OPENAI_COMPATIBLE_@} ${!TARGET_OPENAI_COMPATIBLE_@}; do
    unset "$key"
  done
  export OPENAI_COMPATIBLE_BASE_URL=https://tokenhub.sensetime.com/v1
  export OPENAI_COMPATIBLE_MODEL=deepseek-v4.1-flash-1017/huoshan_L/sfc
  export OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS=131072
  export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS=131072
  export SKILLOPT_WORKERS=24 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
  tag=${OPENAI_COMPATIBLE_MODEL//\//-}

  while read -r bench arm first_dir; do
    session="so2-$stamp-$bench-$arm"
    out="$repo/outputs/skillopt_${bench}_${tag}_${stamp}_rep2_${arm}"
    args=("$py" -B -u scripts/train.py --config "configs/$bench/default.yaml"
          --cfg-options env.workers=24 env.max_completion_tokens=131072
          model.backend=openai_compatible model.optimizer_backend=openai_compatible
          model.target_backend=openai_compatible "model.optimizer=$OPENAI_COMPATIBLE_MODEL"
          "model.target=$OPENAI_COMPATIBLE_MODEL" "env.out_root=$out")
    if [[ $arm != fullval ]]; then
      args+=(evaluation.selection_early_stop=true evaluation.selection_early_stop_batch=4)
    fi
    case "$arm" in
      agg_b4_rho075) args+=(evaluation.train_early_stop=true evaluation.train_early_stop_rho=0.75) ;;
      agg_b4_rho050) args+=(evaluation.train_early_stop=true evaluation.train_early_stop_rho=0.5) ;;
    esac
    if [[ $bench == searchqa && $arm == agg_b4_rho075 ]]; then
      args+=(env.eval-ignore-this=1)  # 保留该组首轮配置中的原字段
    fi

    mkdir "$out"  # 目录已存在时退出，保留既有结果
    export RUN_DIR="$out" FIRST_RUN="$repo/outputs/$first_dir" REVISION="$revision"
    tmux -L "$session" -f /dev/null new-session -d -s "$session" -c "$repo" bash -c '
      set -eu
      umask 077
      exec >>"$RUN_DIR/console.log" 2>&1
      finish() { rc=$?; printf "%s\n" "$rc" > "$RUN_DIR/exit.code"; date -Is > "$RUN_DIR/finished.at"; }
      trap finish EXIT
      date -Is | tee "$RUN_DIR/started.at"
      printf "first_run=%s\nrevision=%s\nworkers=%s\n" "$FIRST_RUN" "$REVISION" "$SKILLOPT_WORKERS"
      printf "argv:"; printf " %q" "$@"; printf "\n"
      "$@"
    ' _ "${args[@]}"
    printf '%s -> %s\n' "$session" "$out"
  done <<'RUNS'
searchqa fullval skillopt_searchqa_deepseek-v4-1-flash-huoshan-sfc_20261118_200115_fullval
searchqa earlystop_b4 skillopt_searchqa_deepseek-v4-1-flash-huoshan-sfc_20261118_200115_earlystop_b4
searchqa agg_b4_rho075 skillopt_searchqa_deepseek-v4.1-flash-1017_20261119_211735_agg_b4_rho075
searchqa agg_b4_rho050 skillopt_searchqa_deepseek-v4.1-flash-1017L_20261120_002105_agg_b4_rho050
triviaqa fullval skillopt_triviaqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_fullval
triviaqa earlystop_b4 skillopt_triviaqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_earlystop_b4
triviaqa agg_b4_rho075 skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
triviaqa agg_b4_rho050 skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050
hotpotqa fullval skillopt_hotpotqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_fullval
hotpotqa earlystop_b4 skillopt_hotpotqa_deepseek-v4-1-flash-huoshan-sfc_20261118_103658_earlystop_b4
hotpotqa agg_b4_rho075 skillopt_hotpotqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
hotpotqa agg_b4_rho050 skillopt_hotpotqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050
bird fullval skillopt_bird_deepseek-v4.1-flash-1017_20261118_073747_fullval
bird earlystop_b4 skillopt_bird_deepseek-v4.1-flash-1017_20261118_200644_earlystop_b4
bird agg_b4_rho075 skillopt_bird_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
bird agg_b4_rho050 skillopt_bird_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050
RUNS
)
```

原批次结果位于 `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/`。例如，SearchQA 完整训练的目录名为 `skillopt_searchqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_080455_rep2_fullval/`；对应会话可用 `tmux -L so2-20261126_080455-searchqa-fullval attach -t so2-20261126_080455-searchqa-fullval` 查看。各组的 `console.log` 保留实际 argv、首轮路径与代码版本，`config.json` 保留生效参数，`exit.code` 和 `summary.json` 用于核对完成状态。

## TriviaQA 二次进化首轮运行（2026-11-26，北京时间）

2026-11-26 17:09:26（UTC+8）在 CPU 启动两个并行 tmux 会话，使用代码提交 `dad88a8e00db060af2ce9bab84c26ff6320c6af2`。两组均于 17:09:27 创建结果目录。以下路径已在 CPU 上与各自的 `manifest.json` 核对。

| 配置 / tmux 会话 | 第一阶段来源绝对路径 | 本次二次进化结果绝对路径 | 初始最佳 dev | 每个子代编辑预算 |
|---|---|---|---:|---:|
| `rho050` | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050` | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_rho050` | 174/200 | 3 |
| `rho075` | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075` | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_rho075` | 178/200 | 2 |

每组只读自己的首轮轨迹，在 200 道 dev 题上验证。每组初始 5 个个体，使用 `matrix` 选供体，种子 42，最多 3 代，连续 2 代无人替换即停。生成和答题均使用归档模型 `deepseek-v4.1-flash-1017/huoshan_L/sfc`，单次请求的输出上限均为 131072。归档 worker 数为 24；每组依次评估子代，dev 早停每批最多并行 4 题。

在 Mac 上执行 `ssh cpu`，再在 CPU 的 Bash 中运行下面的命令即可复现。API key 在提示时输入。新运行会生成新的时间戳、会话和结果目录。

```bash
(
set -eu
umask 077
cd /mnt/sdd3/xuancheng/code/github/SkillOpt
git diff --quiet dad88a8e00db060af2ce9bab84c26ff6320c6af2 -- skillopt skillopt_evolution
export OPENAI_COMPATIBLE_BASE_URL='https://tokenhub.sensetime.com/v1'
export OPENAI_COMPATIBLE_MODEL='deepseek-v4.1-flash-1017/huoshan_L/sfc'
export OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS=131072
export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS=131072
export SKILLOPT_WORKERS=24
unset OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL TARGET_OPENAI_COMPATIBLE_BASE_URL
unset OPTIMIZER_OPENAI_COMPATIBLE_API_KEY TARGET_OPENAI_COMPATIBLE_API_KEY
read -rsp 'TokenHub API key: ' OPENAI_COMPATIBLE_API_KEY
echo
: "${OPENAI_COMPATIBLE_API_KEY:?TokenHub key required}"
export OPENAI_COMPATIBLE_API_KEY
source050=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_111321_agg_b4_rho050
source075=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
sock="trivia-evo2-$stamp"
for arm in rho050 rho075; do
  case "$arm" in
    rho050) src=$source050 ;;
    rho075) src=$source075 ;;
  esac
  tmux -L "$sock" -f /dev/null new-session -d -s "$arm" -c "$PWD" \
    "tmux set-window-option remain-on-exit on && exec .venv/bin/python -B -u -m skillopt_evolution.run --source '$src' --strategy matrix --generations 3 --patience 2 --seed 42"
done
echo "tmux socket=$sock"
tmux -L "$sock" list-sessions
)
```

本次会话可用 `tmux -L trivia-evo2-20261126_170926 attach -t rho050` 或 `-t rho075` 查看。各结果目录的 `generations/` 保存供体、编辑及逐批 dev 记录；`summary.json` 的 `incremental_usage` 按生成和验证列出输入、输出与总 token。中断后可以用 `.venv/bin/python -u -m skillopt_evolution.run --resume <本次二次进化结果绝对路径>` 续跑。

### 同批 TriviaQA 的自动 test 评测（2026-11-26，北京时间）

当前这两个已有运行各自使用独立的收尾进程，等待对应的 `summary.json` 标记为 complete，核对并冻结二次进化选定的 skill，然后使用该运行的答题模型、131072 输出上限和 24 个 workers 跑固定 1,000 道 test。即使最终 skill 与第一阶段相同，也进行一次新推理。每组结果在上表对应的结果目录下的 `test/`，可查看 `status.json`、`manifest.json`、逐题 `results.jsonl` 和最终 `summary.json`（EM、F1、失败数、单独的 test 输入/输出 token 及相对第一阶段的变化）。

在 CPU 仓库根目录，这批任务的挂接命令如下。已有 tmux 服务端继承了本批的模型凭据。更换服务器或新开 shell 时需设置同一服务的 URL 和 key。

```bash
cd /mnt/sdd3/xuancheng/code/github/SkillOpt
sock=trivia-evo2-20261126_170926
for arm in rho050 rho075; do
  dir="$PWD/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_$arm"
  tmux -L "$sock" new-session -d -s "test-$arm" -c "$PWD" \
    "tmux set-window-option remain-on-exit on && exec .venv/bin/python -B -u -m scripts.evolution_test.run --evolution-dir '$dir'"
done
```

今后新建一组 TriviaQA 运行，使用 `.venv/bin/python -u -m scripts.evolution_test.pipeline --source <第一阶段目录> --strategy matrix`，进化进程正常结束后自动评估该次运行的最终 skill。若 test 被中断，可单独用 `.venv/bin/python -u -m scripts.evolution_test.run --evolution-dir <该次二次进化目录>` 续评；已经成功作答的题直接复用。

### TriviaQA `p=0.75` 固定供体提示词对照（2026-11-26，北京时间）

17:57:28 在 CPU 的独立 tmux 会话 `prompt-replay-rho075-20261126_175728` 启动，运行提交为 `5387d4c1d25e6cfffd6cf23f1da3d377aba7ded2`。输入为 `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_rho075`，本次实际输出为 `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_175731_096490_evo2_prompt_replay_rho075`。

在 CPU 仓库根目录、同一 TokenHub API 环境下，用以下命令复现。新运行会创建同级的新时间戳目录。固定 10 个位置的父本、供体、dev 题序及目标/反例来自旧运行。结果在新目录的 `summary.json`，按实际生成、明确跳过及固定分母分别统计；每个位置的 `comparison.json` 保留答案、分数、成本和来源。

```bash
cd /mnt/sdd3/xuancheng/code/github/SkillOpt
prior=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_170927_evo2_rho075
.venv/bin/python -B -u -m skillopt_evolution.replay_prompt --prior-run "$prior"
```

上述固定供体对照已按要求停止，保留已有日志。下一组从第一阶段独立启动完整的第二阶段。

### TriviaQA 新提示词完整二次进化，p=0.75（2026-11-26，北京时间）

18:09:51 在 CPU 通过以下命令启动，运行提交为 `d8fc4c263954a30645faf02c38b85ce55ecd3211`。复用首轮 120 步历史，五个初始个体的 dev 正确数分别为 178、177、176、176、175（共 200 题）。每代重新选供体，子代胜过父本即可进入下一代。最多 3 代，连续 2 代无人替换则提前结束。进化正常结束后，pipeline 自动用最终 dev best 评测 1,000 道固定 test 题。

- 第一阶段来源：`/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075`
- 第二阶段实际结果：`/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261126_180954_evo2_rho075`。其中 `summary.json` 记录 dev 结果和进化成本，`test/summary.json` 记录最终 test EM、F1 和推理成本。
- tmux socket 为 `trivia-evo2-20261126_170926`，会话为 `full-newprompt-rho075-20261126_180951`，继承已有模型环境。

```bash
cd /mnt/sdd3/xuancheng/code/github/SkillOpt
source075=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
sock=trivia-evo2-20261126_170926
session="full-newprompt-rho075-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
tmux -L "$sock" new-session -d -s "$session" -c "$PWD" \
  "umask 077; tmux set-window-option remain-on-exit on && exec .venv/bin/python -B -u -m scripts.evolution_test.pipeline --source '$source075' --strategy matrix --generations 3 --patience 2 --seed 42"
```

### TriviaQA 双成功供体完整二次进化，p=0.75（2026-11-27，北京时间）

CPU 于 01:20:19 从 Git 提交 `822c35ee100145fdafe88850a30908291d37a41c` 启动完整二次进化。来源为首轮 120 步归档，五个父本 dev 得分为 178、177、176、176、175（共 200 题）。本次 A、B 必须共同答对 X 的错题；生成器读取当时的原始文段和实际可见作答，引用 A/B 共同规则后与 X 做差分。不合适的供体和重复文本触发补位，每个完成的代必须有 5 个不同的新子代实际完成 dev 选优。种子 42，最多 3 代，连续 2 个完成的代无人超过父本时停止。进化成功后同一 pipeline 自动使用最终 best 评固定 1,000 道 test。

| 项目 | 已核对的 CPU 路径或标识 |
|---|---|
| 第一阶段来源 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075` |
| 本次第二阶段 | `/mnt/sdd3/xuancheng/code/github/SkillOpt/outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017-huoshan_L-sfc_20261127_012020_evo2_rho075` |
| tmux | socket `trivia-evo2-20261126_170926`，会话 `dual-success-rho075-20261127_012019` |

在 CPU 仓库根目录使用同一模型服务环境复现。程序为新运行分配新的时间戳，输出上限为 131072。进化成本记在结果目录的 `summary.json`；最终 EM、F1 和单独的 test token 成本记在 `test/summary.json`。

```bash
cd /mnt/sdd3/xuancheng/code/github/SkillOpt
source075=outputs/skillopt_triviaqa_deepseek-v4.1-flash-1017L_20261120_001733_agg_b4_rho075
sock=trivia-evo2-20261126_170926
session="dual-success-rho075-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
tmux -L "$sock" new-session -d -s "$session" -c "$PWD" \
  "umask 077; tmux set-window-option remain-on-exit on && exec .venv/bin/python -B -u -m scripts.evolution_test.pipeline --source '$source075' --strategy matrix --generations 3 --patience 2 --seed 42"
```
