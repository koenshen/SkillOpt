#!/usr/bin/env bash
# Launch selection-early-stop (batch=4) vs full-validation training pairs for
# OfficeQA, LiveMathematicianBench and SpreadsheetBench, one tmux session per
# run.  All other settings match the 261112 SearchQA earlystop/fullval pair.
#
# Before running, export in the launching shell (keys are never stored here):
#   export OPENAI_COMPATIBLE_API_KEY="sk-xxx"  # tokenhub key
#   export EMBEDDING_API_KEY="sk-xxx"          # siliconflow key (bge-m3)
# Everything else has a default below and can be overridden by exporting it.
# Every variable is handed to each session with `tmux new-session -e`, because
# tmux does not copy them from the launching shell into a new session.
#
# Optional: pass bench names to launch only those, e.g.
#   bash scripts/launch_earlystop_b4.sh officeqa livemath
#
# Session names: <YYMMDD>-skillopt-<bench>-dpskv41flash-train-{earlystop-b4,fullval}
# Outputs:       outputs/skillopt_<bench>_deepseek-v4-1-flash-huoshan-huoshan_<TS>_{earlystop_b4,fullval}
set -euo pipefail
: "${OPENAI_COMPATIBLE_API_KEY:?export OPENAI_COMPATIBLE_API_KEY first}"
: "${EMBEDDING_API_KEY:?export EMBEDDING_API_KEY first}"

export OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS="${OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS:-131072}"
export TARGET_OPENAI_COMPATIBLE_MAX_TOKENS="${TARGET_OPENAI_COMPATIBLE_MAX_TOKENS:-131072}"
export SKILLOPT_WORKERS="${SKILLOPT_WORKERS:-4}"  # read by test-phase only; train uses env.workers below
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-https://api.siliconflow.cn/v1/embeddings}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
export OPENAI_COMPATIBLE_BASE_URL="${OPENAI_COMPATIBLE_BASE_URL:-https://tokenhub.sensetime.com/v1}"
export OPENAI_COMPATIBLE_MODEL="${OPENAI_COMPATIBLE_MODEL:-deepseek-v4.1-flash-0910/huoshan/huoshan}"

# Show only length + head/tail so a truncated or masked key is easy to spot.
for v in OPENAI_COMPATIBLE_API_KEY EMBEDDING_API_KEY; do
  val="${!v}"
  echo "$v len=${#val} head=${val:0:6} tail=${val: -4}"
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
TS=$(date +%Y%m%d_%H%M%S)
D=$(date +%y%m%d)
M="$OPENAI_COMPATIBLE_MODEL"
TAG="${RUN_TAG:-deepseek-v4-1-flash-huoshan-huoshan}"  # same output-dir naming as the 261112 SearchQA pair
COMMON="env.workers=24 env.max_completion_tokens=$TARGET_OPENAI_COMPATIBLE_MAX_TOKENS model.backend=openai_compatible model.optimizer_backend=openai_compatible model.target_backend=openai_compatible model.optimizer=$M model.target=$M"
ES="evaluation.selection_early_stop=true evaluation.selection_early_stop_batch=4"
OQ="env.data_dirs=data/raw/officeqa/treasury_bulletins_parsed/transformed"
echo "code: $(git -C "$REPO" log --oneline -1)  ts=$TS"

ENVARGS=()
for v in OPTIMIZER_OPENAI_COMPATIBLE_MAX_TOKENS TARGET_OPENAI_COMPATIBLE_MAX_TOKENS SKILLOPT_WORKERS \
         EMBEDDING_BASE_URL EMBEDDING_API_KEY EMBEDDING_MODEL \
         OPENAI_COMPATIBLE_BASE_URL OPENAI_COMPATIBLE_API_KEY OPENAI_COMPATIBLE_MODEL; do
  ENVARGS+=(-e "$v=${!v}")
done

launch() {  # <session> <config> <extra cfg-options> <out suffix> <bench dir name>
  local name=$1 cfg=$2 extra=$3 suffix=$4 bench=$5
  local out="outputs/skillopt_${bench}_${TAG}_${TS}_${suffix}"
  if tmux has-session -t "=$name" 2>/dev/null; then echo "exists: $name"; return; fi
  # The job runs as the session's command; `exec bash` keeps the pane open
  # afterwards so the output stays visible after the job ends.
  tmux new-session -d -s "$name" -c "$REPO" "${ENVARGS[@]}" \
    "source .venv/bin/activate && python scripts/train.py --config $cfg --cfg-options $COMMON $extra env.out_root=$out; exec bash"
  local k
  k=$(tmux show-environment -t "=$name" OPENAI_COMPATIBLE_API_KEY | cut -d= -f2-)
  echo "started: $name -> $out (session key len=${#k} tail=${k: -4})"
}

SELECTED=("$@")
sel() {  # true if no bench was named on the command line, or $1 was named
  [ ${#SELECTED[@]} -eq 0 ] && return 0
  local b; for b in "${SELECTED[@]}"; do [ "$b" = "$1" ] && return 0; done
  return 1
}

if sel officeqa; then
  launch "$D-skillopt-officeqa-dpskv41flash-train-earlystop-b4" configs/officeqa/default.yaml "$OQ $ES" earlystop_b4 officeqa
  launch "$D-skillopt-officeqa-dpskv41flash-train-fullval"      configs/officeqa/default.yaml "$OQ"     fullval      officeqa
fi
if sel livemath; then
  launch "$D-skillopt-livemath-dpskv41flash-train-earlystop-b4" configs/livemathematicianbench/default.yaml "$ES" earlystop_b4 livemathematicianbench
  launch "$D-skillopt-livemath-dpskv41flash-train-fullval"      configs/livemathematicianbench/default.yaml ""    fullval      livemathematicianbench
fi
if sel spreadsheetbench; then
  launch "$D-skillopt-spreadsheetbench-dpskv41flash-train-earlystop-b4" configs/spreadsheetbench/default.yaml "$ES" earlystop_b4 spreadsheetbench
  launch "$D-skillopt-spreadsheetbench-dpskv41flash-train-fullval"      configs/spreadsheetbench/default.yaml ""    fullval      spreadsheetbench
fi
