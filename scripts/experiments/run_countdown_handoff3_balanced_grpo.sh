#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_MODEL="${BASE_MODEL:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
INITIAL_ADAPTER="${INITIAL_ADAPTER:-outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final}"
SFT_JSONL="${SFT_JSONL:-outputs/experiments/handoff3_countdown_balanced_sft/data/countdown-balanced-low-solution-sft-20000.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json}"

ROOT="${ROOT:-outputs/experiments/handoff3_countdown_balanced_grpo_probe}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${ROOT}/data/countdown-balanced-grpo-train-manifest.json}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT}/rollout_train256_g8_targetdistance_512}"
POOL_MANIFEST="${POOL_MANIFEST:-${ROOT}/pool_train256_g8_targetdistance_512/pool-manifest.json}"
TRAIN_DIR="${TRAIN_DIR:-${ROOT}/grpo_targetdistance_from_balanced_sft_train256_g8_lr1e6_beta001_1200}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/eval_targetdistance_1200_full100_4096}"
AUDIT_DIR="${AUDIT_DIR:-${ROOT}/rollout_targetdistance_1200_full100_g8_4096}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"

mkdir -p "${ROOT}/data" "${LOG_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(timestamp)] $*"
}

log "Countdown handoff3 balanced GRPO started"
log "base model: ${BASE_MODEL}"
log "initial adapter: ${INITIAL_ADAPTER}"
log "sft jsonl: ${SFT_JSONL}"
log "train manifest: ${TRAIN_MANIFEST}"
log "train dir: ${TRAIN_DIR}"

if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
  log "building balanced GRPO train manifest"
  "${PYTHON_BIN}" scripts/build_countdown_grpo_manifest_from_sft_jsonl.py \
    --sft-jsonl "${SFT_JSONL}" \
    --output "${TRAIN_MANIFEST}" \
    --manifest-split countdown_grpo_train \
    --seed 20260620 \
    --bucket-quotas 1=64 2=64 3=64 4=64 \
    >"${LOG_DIR}/00_build_grpo_manifest.log" 2>&1
else
  log "balanced GRPO train manifest already exists"
fi

if [[ ! -f "${ROLLOUT_DIR}/summary.json" ]]; then
  log "running train-side rollout audit for GRPO pool"
  mkdir -p "${ROLLOUT_DIR}"
  "${PYTHON_BIN}" scripts/experiments/audit_rollout_distribution.py \
    --manifest "${TRAIN_MANIFEST}" \
    --split countdown_grpo_train \
    --checkpoint "${INITIAL_ADAPTER}" \
    --model-name "${BASE_MODEL}" \
    --training-mode lora \
    --prompt-style qwen_chat_minimal_target \
    --output-dir "${ROLLOUT_DIR}" \
    --num-generations 8 \
    --batch-size 4 \
    --max-new-tokens 512 \
    --temperature 0.8 \
    --top-p 0.95 \
    --reward-profile target_distance \
    --seed 20260620 \
    >"${LOG_DIR}/01_rollout_train256_g8_targetdistance.log" 2>&1
else
  log "train-side rollout audit already exists"
fi

if [[ ! -f "${POOL_MANIFEST}" ]]; then
  log "building GRPO pool from reward-variance groups"
  mkdir -p "$(dirname "${POOL_MANIFEST}")"
  "${PYTHON_BIN}" scripts/build_grpo_pool.py \
    --details "${ROLLOUT_DIR}/countdown_grpo_train-rollout-details.json" \
    --output "${POOL_MANIFEST}" \
    --split countdown_grpo_train \
    --checkpoint "${INITIAL_ADAPTER}" \
    --seed 20260620 \
    --num-generations 8 \
    --temperature 0.8 \
    --top-p 0.95 \
    --max-new-tokens 512 \
    --min-pool-size 1 \
    --min-mixed-group-rate 0.0 \
    --max-zero-std-group-rate 1.0 \
    --min-correct-truncation-mixed 0 \
    --max-all-wrong-rate 1.0 \
    >"${LOG_DIR}/02_build_pool.log" 2>&1
else
  log "GRPO pool already exists"
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
pool = json.loads(Path("${POOL_MANIFEST}").read_text())
selected = pool.get("selected_prompt_ids", [])
print({"selected_prompt_ids": len(selected), "pool": "${POOL_MANIFEST}"})
if not selected:
    raise SystemExit("GRPO pool has no selected prompt ids")
PY

COMMON_GRPO=(
  "${PYTHON_BIN}" scripts/train_grpo.py
  --manifest "${TRAIN_MANIFEST}"
  --split countdown_grpo_train
  --output-dir "${TRAIN_DIR}"
  --prompt-style qwen_chat_minimal_target
  --model-name-or-path "${BASE_MODEL}"
  --initial-adapter "${INITIAL_ADAPTER}"
  --pool-manifest "${POOL_MANIFEST}"
  --max-steps 1200
  --save-steps 300
  --logging-steps 10
  --max-completion-length 512
  --num-generations 8
  --gradient-accumulation-steps 8
  --temperature 0.8
  --top-p 0.95
  --learning-rate 1e-6
  --beta 0.001
  --scale-rewards none
  --reward-profile target_distance
  --peft-mode lora
  --no-mask-truncated-completions
  --no-remove-unused-columns
)

if [[ ! -f "${TRAIN_DIR}/compat-probe.json" ]]; then
  log "running GRPO compatibility probe"
  mkdir -p "${TRAIN_DIR}"
  "${COMMON_GRPO[@]}" --compat-probe >"${LOG_DIR}/03_compat_probe.log" 2>&1
else
  log "GRPO compatibility probe already exists"
fi

if [[ ! -f "${TRAIN_DIR}/final/adapter_model.safetensors" ]]; then
  log "training GRPO adapter"
  "${COMMON_GRPO[@]}" --train >"${LOG_DIR}/04_train_grpo.log" 2>&1
else
  log "GRPO adapter already exists"
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
  >"${LOG_DIR}/05_eval_full100_4096.log" 2>&1

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
  --reward-profile target_distance \
  --seed 20260620 \
  >"${LOG_DIR}/06_audit_full100_g8_4096.log" 2>&1

log "Countdown handoff3 balanced GRPO finished"
