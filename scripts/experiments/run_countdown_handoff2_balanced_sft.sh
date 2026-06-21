#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_MODEL="${BASE_MODEL:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-outputs/experiments/handoff3_countdown_sft/sft_handoff2_target_replacement_20000_2400steps/final}"
COUNTDOWN_JSONL="${COUNTDOWN_JSONL:-data/raw/hf/Jiayi-Pan__Countdown-Tasks-3to4/default__train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json}"

ROOT="${ROOT:-outputs/experiments/handoff3_countdown_balanced_sft}"
TRAIN_JSONL="${TRAIN_JSONL:-${ROOT}/data/countdown-balanced-low-solution-sft-20000.jsonl}"
TRAIN_META="${TRAIN_META:-${ROOT}/data/countdown-balanced-low-solution-sft-20000.metadata.json}"
TRAIN_DIR="${TRAIN_DIR:-${ROOT}/sft_from_target_replacement_balanced_low_solution_20000_2400steps}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/eval_balanced_low_solution_20000_2400steps_full100_4096}"
AUDIT_DIR="${AUDIT_DIR:-${ROOT}/rollout_balanced_low_solution_20000_2400steps_full100_g8_4096}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"

mkdir -p "${ROOT}/data" "${LOG_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(timestamp)] $*"
}

log "Countdown balanced low-solution SFT started"
log "base model: ${BASE_MODEL}"
log "initial adapter: ${INITIAL_ADAPTER}"
log "train jsonl: ${TRAIN_JSONL}"
log "train dir: ${TRAIN_DIR}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  log "building balanced SFT data"
  "${PYTHON_BIN}" scripts/build_countdown_sft_data.py \
    --local-jsonl "${COUNTDOWN_JSONL}" \
    --exclude-manifest "${EVAL_MANIFEST}" \
    --exclude-split countdown_eval \
    --output "${TRAIN_JSONL}" \
    --metadata-output "${TRAIN_META}" \
    --max-scan-rows 490364 \
    --traces-per-puzzle 8 \
    --prompt-style qwen_chat_minimal_target \
    --trace-type short_success \
    --min-numbers 3 \
    --max-numbers 4 \
    --solution-cap 5 \
    --bucket-quotas 1=5000 2=5000 3=5000 4=5000 \
    >"${LOG_DIR}/build_balanced_sft_data.log" 2>&1
else
  log "balanced SFT data already exists"
fi

if [[ ! -f "${TRAIN_DIR}/final/adapter_model.safetensors" ]]; then
  log "training balanced SFT adapter"
  mkdir -p "${TRAIN_DIR}"
  "${PYTHON_BIN}" scripts/experiments/run_countdown_sft_adaptation.py \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${TRAIN_DIR}" \
    --base-model "${BASE_MODEL}" \
    --initial-adapter "${INITIAL_ADAPTER}" \
    --max-length 4096 \
    --max-steps 2400 \
    --save-steps 600 \
    --logging-steps 40 \
    --learning-rate 1e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --seed 20260620 \
    >"${LOG_DIR}/train_balanced_sft.log" 2>&1
else
  log "balanced SFT adapter already exists"
fi

log "evaluating full100 greedy"
mkdir -p "${EVAL_DIR}"
"${PYTHON_BIN}" scripts/eval_checkpoint.py \
  --manifest "${EVAL_MANIFEST}" \
  --split countdown_eval \
  --output-dir "${EVAL_DIR}" \
  --model-name "${BASE_MODEL}" \
  --checkpoint "${TRAIN_DIR}/final" \
  --training-mode lora \
  --batch-size 4 \
  --max-new-tokens 4096 \
  --prompt-style qwen_chat_minimal_target \
  >"${LOG_DIR}/eval_full100_4096.log" 2>&1

log "running full100 sampled audit"
mkdir -p "${AUDIT_DIR}"
"${PYTHON_BIN}" scripts/experiments/audit_rollout_distribution.py \
  --manifest "${EVAL_MANIFEST}" \
  --split countdown_eval \
  --limit 100 \
  --checkpoint "${TRAIN_DIR}/final" \
  --model-name "${BASE_MODEL}" \
  --training-mode lora \
  --prompt-style qwen_chat_minimal_target \
  --output-dir "${AUDIT_DIR}" \
  --num-generations 8 \
  --batch-size 4 \
  --max-new-tokens 4096 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 20260620 \
  >"${LOG_DIR}/audit_full100_g8_4096.log" 2>&1

log "Countdown balanced low-solution SFT finished"
