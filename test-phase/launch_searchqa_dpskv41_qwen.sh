#!/usr/bin/env bash
# Launch every missing SearchQA test-phase job for deepseek-v4-1-flash and
# Qwen3.8-27B (top_k / best_repeat x k=3,5,7, each followed by
# rag_plus_history_show_self), one tmux session per job.
#
# Before running, export in the launching shell:
#   export OPENAI_COMPATIBLE_API_KEY="sk-xxx"  # tokenhub key (deepseek-v4-1-flash)
#   export EMBEDDING_API_KEY="sk-xxx"          # siliconflow key (bge-m3)
# The keys are handed to each tmux session with `tmux new-session -e`;
# run_searchqa_job.sh then sets base URL / model (and "empty" key for Qwen) per session.
#
# Optional: pass "dpsk" or "qwen" to launch only that model's jobs.
#
# Session names: <YYMMDD-HHMM>-skillopt-searchqa-<model>-<mode>-<k>-showself
# Logs:          logs/<session>.infer.log, logs/<session>.showself.log
set -euo pipefail
: "${OPENAI_COMPATIBLE_API_KEY:?export OPENAI_COMPATIBLE_API_KEY first}"
: "${EMBEDDING_API_KEY:?export EMBEDDING_API_KEY first}"

ONLY=${1:-all}
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAMP=$(date +%y%m%d-%H%M)
RUN="bash $REPO/test-phase/run_searchqa_job.sh"

launch() {  # <session_name> <args for run_searchqa_job.sh...>
  local name=$1; shift
  if tmux has-session -t "=$name" 2>/dev/null; then echo "exists: $name"; return; fi
  tmux new-session -d -s "$name" -c "$REPO" \
    -e "OPENAI_COMPATIBLE_API_KEY=$OPENAI_COMPATIBLE_API_KEY" -e "EMBEDDING_API_KEY=$EMBEDDING_API_KEY"
  tmux send-keys -t "=$name" "$RUN $* $name" Enter
  echo "started: $name"
}

if [ "$ONLY" = all ] || [ "$ONLY" = dpsk ]; then
  # deepseek-v4-1-flash: top_k k=5/7 inference already exists -> show_self only
  launch "$STAMP-skillopt-searchqa-dpskv41flash-topk-5-showself" dpskv41flash existing \
    outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-huoshan_20261010_062054_top_k_numskills5
  launch "$STAMP-skillopt-searchqa-dpskv41flash-topk-7-showself" dpskv41flash existing \
    outputs/skillopt_searchqa_deepseek-v4-1-flash-huoshan-huoshan_20261010_070125_top_k_numskills7
  launch "$STAMP-skillopt-searchqa-dpskv41flash-topk-3-showself" dpskv41flash top_k 3
  for k in 3 5 7; do
    launch "$STAMP-skillopt-searchqa-dpskv41flash-bestrepeat-$k-showself" dpskv41flash best_repeat "$k"
  done
fi

if [ "$ONLY" = all ] || [ "$ONLY" = qwen ]; then
  # Qwen3.8-27B: nothing run yet
  for k in 3 5 7; do
    launch "$STAMP-skillopt-searchqa-qwen3827b-topk-$k-showself" qwen3827b top_k "$k"
    launch "$STAMP-skillopt-searchqa-qwen3827b-bestrepeat-$k-showself" qwen3827b best_repeat "$k"
  done
fi
