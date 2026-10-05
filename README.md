# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

## 261003 self command - install venv
```bash
uv venv --python 3.11.15
source .venv/bin/activate
uv pip install -r requirements.txt

export OPENAI_COMPATIBLE_BASE_URL="https://tokenhub.sensetime.com/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-xxx"
export OPENAI_COMPATIBLE_MODEL="bailian/deepseek-v4-flash-0731"

export EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1/embeddings"
export EMBEDDING_API_KEY="sk-xxx"
export EMBEDDING_MODEL="BAAI/bge-m3"
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
```

## 261003 self command - train and test
```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=bailian/deepseek-v4-flash-0731 \
    model.target=bailian/deepseek-v4-flash-0731
    
export ALFWORLD_DATA="$HOME/.cache/alfworld"
python scripts/train.py \
  --config configs/alfworld/default.yaml \
  --cfg-options \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=bailian/deepseek-v4-flash-0731 \
    model.target=bailian/deepseek-v4-flash-0731

python scripts/train.py \
  --config configs/spreadsheetbench/default.yaml \
  --cfg-options \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=bailian/deepseek-v4-flash-0731 \
    model.target=bailian/deepseek-v4-flash-0731
    
python scripts/train.py \
  --config configs/livemathematicianbench/default.yaml \
  --cfg-options \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=bailian/deepseek-v4-flash-0731 \
    model.target=bailian/deepseek-v4-flash-0731
    
python scripts/train.py \
  --config configs/officeqa/default.yaml \
  --cfg-options \
    env.data_dirs=data/raw/officeqa/treasury_bulletins_parsed/transformed \
    model.backend=openai_compatible \
    model.optimizer_backend=openai_compatible \
    model.target_backend=openai_compatible \
    model.optimizer=bailian/deepseek-v4-flash-0731 \
    model.target=bailian/deepseek-v4-flash-0731
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
```