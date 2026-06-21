#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_MODEL="${BASE_MODEL:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
HANDOFF2_ADAPTER="${HANDOFF2_ADAPTER:-outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/train/checkpoint-500}"
EVAL_MANIFEST="${EVAL_MANIFEST:-outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json}"
BALANCED16_MANIFEST="${BALANCED16_MANIFEST:-outputs/experiments/handoff3_countdown_adapt/countdown-balanced16-eval-manifest.json}"
FIVEPLUS_JSONL="${FIVEPLUS_JSONL:-outputs/experiments/handoff3_countdown_adapt/sft/countdown-solver-sft-8000-5plus.jsonl}"
MIXED_JSONL="${MIXED_JSONL:-outputs/experiments/handoff3_countdown_adapt/sft/countdown-solver-sft-20000-mixed.jsonl}"

ROOT="${ROOT:-outputs/experiments/handoff3_countdown_adapt}"
STAGE1_DIR="${STAGE1_DIR:-${ROOT}/sft_train_handoff2_curriculum_5plus8000_1200steps}"
STAGE2_DIR="${STAGE2_DIR:-${ROOT}/sft_train_handoff2_curriculum_5plus8000_1200_then_mixed20000_2400steps}"
LOG_DIR="${LOG_DIR:-${ROOT}/overnight_curriculum_logs}"

mkdir -p "${LOG_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(timestamp)] $*"
}

wait_for_adapter() {
  local adapter_dir="$1"
  local label="$2"
  while [[ ! -f "${adapter_dir}/final/adapter_model.safetensors" ]]; do
    if ! pgrep -f "${adapter_dir}" >/dev/null 2>&1; then
      log "ERROR: ${label} is not running and final adapter is missing: ${adapter_dir}/final"
      return 1
    fi
    log "waiting for ${label}: ${adapter_dir}/final"
    sleep 300
  done
  log "${label} final adapter ready: ${adapter_dir}/final"
}

eval_balanced16() {
  local checkpoint="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" scripts/eval_checkpoint.py \
    --manifest "${BALANCED16_MANIFEST}" \
    --split countdown_eval_balanced16 \
    --output-dir "${output_dir}" \
    --model-name "${BASE_MODEL}" \
    --checkpoint "${checkpoint}" \
    --training-mode lora \
    --batch-size 4 \
    --max-new-tokens 4096 \
    --prompt-style qwen_chat_target
}

eval_full100() {
  local checkpoint="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" scripts/eval_checkpoint.py \
    --manifest "${EVAL_MANIFEST}" \
    --split countdown_eval \
    --output-dir "${output_dir}" \
    --model-name "${BASE_MODEL}" \
    --checkpoint "${checkpoint}" \
    --training-mode lora \
    --batch-size 4 \
    --max-new-tokens 4096 \
    --prompt-style qwen_chat_target
}

audit_balanced16() {
  local checkpoint="$1"
  local output_dir="$2"
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" scripts/experiments/audit_rollout_distribution.py \
    --manifest "${BALANCED16_MANIFEST}" \
    --split countdown_eval_balanced16 \
    --output-dir "${output_dir}" \
    --model-name "${BASE_MODEL}" \
    --checkpoint "${checkpoint}" \
    --training-mode lora \
    --prompt-style qwen_chat_target \
    --num-generations 8 \
    --batch-size 4 \
    --max-new-tokens 512 \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 20260620
}

log "overnight Countdown curriculum started"
log "base model: ${BASE_MODEL}"
log "stage1 dir: ${STAGE1_DIR}"
log "stage2 dir: ${STAGE2_DIR}"

if [[ ! -f "${STAGE1_DIR}/final/adapter_model.safetensors" ]]; then
  if pgrep -f "${STAGE1_DIR}" >/dev/null 2>&1; then
    wait_for_adapter "${STAGE1_DIR}" "stage1 5plus SFT"
  else
    log "stage1 missing; starting 5plus SFT"
    mkdir -p "${STAGE1_DIR}"
    "${PYTHON_BIN}" scripts/experiments/run_countdown_sft_adaptation.py \
      --train-jsonl "${FIVEPLUS_JSONL}" \
      --output-dir "${STAGE1_DIR}" \
      --base-model "${BASE_MODEL}" \
      --initial-adapter "${HANDOFF2_ADAPTER}" \
      --max-length 4096 \
      --max-steps 1200 \
      --save-steps 600 \
      --logging-steps 20 \
      --learning-rate 2e-5 \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --seed 20260620
  fi
else
  log "stage1 already complete"
fi

log "evaluating stage1 balanced16"
eval_balanced16 \
  "${STAGE1_DIR}/final" \
  "${ROOT}/eval_curriculum_5plus8000_1200steps_balanced16" \
  >"${LOG_DIR}/eval_stage1_balanced16.log" 2>&1

if [[ ! -f "${STAGE2_DIR}/final/adapter_model.safetensors" ]]; then
  log "starting stage2 mixed SFT"
  mkdir -p "${STAGE2_DIR}"
  "${PYTHON_BIN}" scripts/experiments/run_countdown_sft_adaptation.py \
    --train-jsonl "${MIXED_JSONL}" \
    --output-dir "${STAGE2_DIR}" \
    --base-model "${BASE_MODEL}" \
    --initial-adapter "${STAGE1_DIR}/final" \
    --max-length 4096 \
    --max-steps 2400 \
    --save-steps 600 \
    --logging-steps 40 \
    --learning-rate 1e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --seed 20260620
else
  log "stage2 already complete"
fi

log "evaluating stage2 balanced16"
eval_balanced16 \
  "${STAGE2_DIR}/final" \
  "${ROOT}/eval_curriculum_5plus1200_mixed2400_balanced16" \
  >"${LOG_DIR}/eval_stage2_balanced16.log" 2>&1

log "evaluating stage2 full100"
eval_full100 \
  "${STAGE2_DIR}/final" \
  "${ROOT}/eval_curriculum_5plus1200_mixed2400_full100_4096" \
  >"${LOG_DIR}/eval_stage2_full100.log" 2>&1

log "running stage2 balanced16 sampled audit"
audit_balanced16 \
  "${STAGE2_DIR}/final" \
  "${ROOT}/rollout_curriculum_5plus1200_mixed2400_balanced16_n8" \
  >"${LOG_DIR}/audit_stage2_balanced16_n8.log" 2>&1

log "overnight Countdown curriculum finished"
