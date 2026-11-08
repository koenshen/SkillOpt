#!/usr/bin/env bash
# One SearchQA test-phase job: (optional) per-skill inference, then the
# rag_plus_history_show_self vote.  Started by launch_searchqa_dpskv41_qwen.sh,
# one tmux session per job.
#
# Usage:
#   run_searchqa_job.sh <model_tag> <mode> <k> <job_name>               # inference + show_self
#   run_searchqa_job.sh <model_tag> existing <inference_dir> <job_name>  # show_self only
#
# Required environment (passed in by the launcher):
#   OPENAI_COMPATIBLE_API_KEY  tokenhub key (used by dpskv41flash; qwen3827b overrides it with "empty")
#   EMBEDDING_API_KEY          key for the embedding endpoint used by the RAG step
set -euo pipefail

MODEL_TAG=$1; MODE=$2; ARG=$3; JOB=$4
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"
cd "$REPO"

case "$MODEL_TAG" in
  dpskv41flash)
    TRAIN_ROOT=outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-huoshan_20261008_042358
    export OPENAI_COMPATIBLE_BASE_URL=https://tokenhub.sensetime.com/v1
    export OPENAI_COMPATIBLE_API_KEY=${OPENAI_COMPATIBLE_API_KEY:?OPENAI_COMPATIBLE_API_KEY is not set}
    # Training used deepseek-v4-1-flash/huoshan/huoshan; tokenhub now serves it
    # under this name (the old one returns 503 model_not_found).
    export OPENAI_COMPATIBLE_MODEL=deepseek-v4.1-flash-0910/huoshan/huoshan ;;
  qwen3827b)
    TRAIN_ROOT=outputs/skillopt_searchqa_Qwen3.8-27B_20261008_050556
    export OPENAI_COMPATIBLE_BASE_URL=http://106.75.235.194:6077/v1
    export OPENAI_COMPATIBLE_API_KEY=empty
    export OPENAI_COMPATIBLE_MODEL=Qwen3.8-27B ;;
  *) echo "unknown model tag: $MODEL_TAG" >&2; exit 2 ;;
esac
export EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL:-https://api.siliconflow.cn/v1/embeddings}
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-BAAI/bge-m3}
export EMBEDDING_API_KEY=${EMBEDDING_API_KEY:?EMBEDDING_API_KEY is not set}

source .venv/bin/activate

if [ "$MODE" = existing ]; then
  INFER_DIR=$ARG
else
  INFER_LOG=$LOG_DIR/$JOB.infer.log
  echo "[job] $JOB inference -> $INFER_LOG"
  python test-phase/searchqa_test.py --input-root "$TRAIN_ROOT" --mode "$MODE" --num-skills "$ARG" \
    2>&1 | tee "$INFER_LOG"
  INFER_DIR=$(grep -o 'saved=.*' "$INFER_LOG" | tail -1 | cut -d= -f2-)
  if [ -z "$INFER_DIR" ] || [ ! -f "$INFER_DIR/summary.json" ]; then
    echo "[job] inference did not finish; skipping show_self" >&2; exit 1
  fi
fi

SS_LOG=$LOG_DIR/$JOB.showself.log
echo "[job] $JOB show_self on $INFER_DIR -> $SS_LOG"
python test-phase/test_vote_result.py --input-root "$INFER_DIR" --dataset searchqa \
  --policy rag_plus_history_show_self --gate-root "$TRAIN_ROOT" 2>&1 | tee "$SS_LOG"
echo "[job] $JOB done"
