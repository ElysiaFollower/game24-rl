#!/usr/bin/env bash
set -euo pipefail

# Wait for the handoff1 SFT train evaluation to finish, then start the frozen
# GRPO run. This script is intended to run on AutoDL4090 from the repo root.

POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ROOT="${RUN_ROOT:-outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000}"
WATCH_LOG="${WATCH_LOG:-$RUN_ROOT/logs/watch_train_eval_then_grpo.log}"
TRAIN_EVAL_PID_FILE="${TRAIN_EVAL_PID_FILE:-outputs/experiments/baseline_format_v2_full_5000_from800/eval/train_eval_1024_4096.pid}"
TRAIN_1024_REPORT="${TRAIN_1024_REPORT:-outputs/experiments/baseline_format_v2_full_5000_from800/eval/final_train_1024/train-eval-report.json}"
TRAIN_4096_REPORT="${TRAIN_4096_REPORT:-outputs/experiments/baseline_format_v2_full_5000_from800/eval/final_train_4096/train-eval-report.json}"
RANK_AUDIT="${RANK_AUDIT:-outputs/audits/handoff1_sft_final_train_val_test_by_tot_rank_from_raw.json}"

mkdir -p "$(dirname "$WATCH_LOG")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"
}

train_eval_running() {
  pgrep -f "eval_handoff1_sft_final_train|eval_checkpoint.py.*final_train_1024|eval_checkpoint.py.*final_train_4096" >/dev/null
}

grpo_running() {
  pgrep -f "scripts/train_grpo.py.*handoff1_grpo_closure_control_smooth_v1_2000|run_handoff1_grpo_final.sh" >/dev/null
}

log "watcher started"
while true; do
  if grpo_running; then
    log "GRPO already running; watcher exits"
    exit 0
  fi

  if train_eval_running; then
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true)"
    count_1024="$(test -f "${TRAIN_1024_REPORT%/*}/train-raw-outputs.jsonl" && wc -l < "${TRAIN_1024_REPORT%/*}/train-raw-outputs.jsonl" || echo 0)"
    count_4096="$(test -f "${TRAIN_4096_REPORT%/*}/train-raw-outputs.jsonl" && wc -l < "${TRAIN_4096_REPORT%/*}/train-raw-outputs.jsonl" || echo 0)"
    log "waiting: train eval running; raw_counts 1024=$count_1024 4096=$count_4096; gpu=$gpu"
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ -s "$TRAIN_1024_REPORT" && -s "$TRAIN_4096_REPORT" && -s "$RANK_AUDIT" ]]; then
    log "train eval artifacts complete; starting GRPO"
    nohup bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && bash scripts/experiments/run_handoff1_grpo_final.sh" \
      > "$RUN_ROOT/logs/run_handoff1_grpo_final.nohup.log" 2>&1 &
    echo "$!" > "$RUN_ROOT/grpo.pid"
    log "started GRPO pid=$(cat "$RUN_ROOT/grpo.pid")"
    exit 0
  fi

  if [[ -f "$TRAIN_EVAL_PID_FILE" ]]; then
    log "train eval process not running but required artifacts missing; pid_file=$(cat "$TRAIN_EVAL_PID_FILE" 2>/dev/null || true)"
  else
    log "train eval process not running and pid file missing; waiting for required artifacts"
  fi
  log "missing status: train1024=$(test -s "$TRAIN_1024_REPORT" && echo yes || echo no) train4096=$(test -s "$TRAIN_4096_REPORT" && echo yes || echo no) audit=$(test -s "$RANK_AUDIT" && echo yes || echo no)"
  sleep "$POLL_SECONDS"
done
