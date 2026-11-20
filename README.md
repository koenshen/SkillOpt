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

All 16 planned runs below use the same recipe; only the three items marked **[var]** change.

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
| TriviaQA | AGG rho=0.50 | 1017L | *(running at 2026-11-20 11:14)* | | | | | | | |
| HotpotQA | ES (dev early stop, b=4) | 1017 | 160/160 | - | 79 | 160.50M (28.6%) | 145.94M (31.6%) | 0.6780 | 0.6830 | 21.4h |
| HotpotQA | FULL (baseline) | 1017 | 160/160 | - | 5 | 224.63M | 213.35M | 0.6760 | 0.6780 | 14.6h |
| HotpotQA | AGG rho=0.75 | 1017L | *(running at 2026-11-20 11:14: 86/160)* | | | | | | | |
| HotpotQA | AGG rho=0.50 | 1017L | *(running at 2026-11-20 11:14)* | | | | | | | |
| BIRD | ES (dev early stop, b=4) | 1017 | 84/84 | - | 63 | 136.43M (39.4%) | 120.89M (42.2%) | 0.7180 | 0.7380 | 23.6h |
| BIRD | FULL (baseline) | 1017 | 84/84 | - | 45 | 225.14M | 209.21M | 0.7100 | 0.7000 | 11.4h |
| BIRD | AGG rho=0.75 | 1017L | *(running at 2026-11-20 11:14: 38/84)* | | | | | | | |
| BIRD | AGG rho=0.50 | 1017L | *(running at 2026-11-20 11:14)* | | | | | | | |

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
  would explain. Sampling noise on one test split was measured separately: three
  SearchQA runs (2026-11-11/12, `deepseek-v4.1-flash-0910`) that all started
  from the identical initial skill scored 0.7671 / 0.7600 / 0.7571 on the same
  1,400 items, i.e. 0.29-1.00 pt of purely sampling-driven spread.

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
