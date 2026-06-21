#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

BASE_MODEL="${BASE_MODEL:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
BALANCED_ADAPTER="${BALANCED_ADAPTER:-outputs/experiments/handoff3_countdown_balanced_sft/sft_from_target_replacement_balanced_low_solution_20000_2400steps/final}"
COUNTDOWN_JSONL="${COUNTDOWN_JSONL:-data/raw/hf/Jiayi-Pan__Countdown-Tasks-3to4/default__train.jsonl}"
EVAL_MANIFEST="${EVAL_MANIFEST:-outputs/experiments/handoff3_countdown_eval/countdown-stratified-eval-manifest.json}"

ROOT="${ROOT:-outputs/experiments/handoff3_countdown_final_chance}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
DATA_DIR="${DATA_DIR:-${ROOT}/data}"

OVERFIT_JSONL="${OVERFIT_JSONL:-${DATA_DIR}/countdown-overfit128-from-balanced.jsonl}"
OVERFIT_MANIFEST="${OVERFIT_MANIFEST:-${DATA_DIR}/countdown-overfit128-manifest.json}"
OVERFIT_DIR="${OVERFIT_DIR:-${ROOT}/sft_overfit128_from_balanced_1200steps_lr5e5}"
OVERFIT_EVAL_DIR="${OVERFIT_EVAL_DIR:-${ROOT}/eval_overfit128_train_4096}"

UNIQUE_JSONL="${UNIQUE_JSONL:-${DATA_DIR}/countdown-unique-low-solution-sft-20000.jsonl}"
UNIQUE_META="${UNIQUE_META:-${DATA_DIR}/countdown-unique-low-solution-sft-20000.metadata.json}"
UNIQUE_DIR="${UNIQUE_DIR:-${ROOT}/sft_unique20k_from_balanced_6000steps_lr1e5}"
UNIQUE_EVAL_DIR="${UNIQUE_EVAL_DIR:-${ROOT}/eval_unique20k_full100_4096}"
UNIQUE_AUDIT_DIR="${UNIQUE_AUDIT_DIR:-${ROOT}/rollout_unique20k_full100_g8_4096}"
UNIQUE_BUCKET_QUOTAS="${UNIQUE_BUCKET_QUOTAS:-1=5000 2=5000 3=5000 4=5000}"
UNIQUE_MAX_STEPS="${UNIQUE_MAX_STEPS:-6000}"
UNIQUE_SAVE_STEPS="${UNIQUE_SAVE_STEPS:-1500}"
UNIQUE_LEARNING_RATE="${UNIQUE_LEARNING_RATE:-1e-5}"

mkdir -p "${ROOT}" "${DATA_DIR}" "${LOG_DIR}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(timestamp)] $*"
}

log "Countdown final-chance run started"
log "base model: ${BASE_MODEL}"
log "balanced adapter: ${BALANCED_ADAPTER}"
log "root: ${ROOT}"
log "unique bucket quotas: ${UNIQUE_BUCKET_QUOTAS}"

if [[ ! -f "${OVERFIT_JSONL}" ]]; then
  log "building overfit128 JSONL"
  "${PYTHON_BIN}" - <<PY >"${LOG_DIR}/00_build_overfit128.log" 2>&1
import json
import random
from collections import defaultdict
from pathlib import Path

source = Path("outputs/experiments/handoff3_countdown_balanced_sft/data/countdown-balanced-low-solution-sft-20000.jsonl")
output = Path("${OVERFIT_JSONL}")
rng = random.Random(20260621)
by_key = {}
for line in source.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    key = (row["target"], tuple(row["numbers"]))
    by_key.setdefault(key, row)

by_bucket = defaultdict(list)
for row in by_key.values():
    by_bucket[str(row["solution_count_bucket"])].append(row)

selected = []
for bucket in ["1", "2", "3", "4"]:
    rows = sorted(by_bucket[bucket], key=lambda item: item["id"])
    rng.shuffle(rows)
    selected.extend(sorted(rows[:32], key=lambda item: item["id"]))

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as file:
    for row in selected:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

print(json.dumps({
    "source_unique": len(by_key),
    "selected": len(selected),
    "selected_by_bucket": {bucket: 32 for bucket in ["1", "2", "3", "4"]},
}, ensure_ascii=False, indent=2))
PY
else
  log "overfit128 JSONL already exists"
fi

if [[ ! -f "${OVERFIT_MANIFEST}" ]]; then
  log "building overfit128 manifest"
  "${PYTHON_BIN}" scripts/build_countdown_grpo_manifest_from_sft_jsonl.py \
    --sft-jsonl "${OVERFIT_JSONL}" \
    --output "${OVERFIT_MANIFEST}" \
    --manifest-split countdown_overfit128 \
    --seed 20260621 \
    --bucket-quotas 1=32 2=32 3=32 4=32 \
    >"${LOG_DIR}/01_build_overfit128_manifest.log" 2>&1
else
  log "overfit128 manifest already exists"
fi

if [[ ! -f "${OVERFIT_DIR}/final/adapter_model.safetensors" ]]; then
  log "training overfit128 sanity adapter"
  "${PYTHON_BIN}" scripts/experiments/run_countdown_sft_adaptation.py \
    --train-jsonl "${OVERFIT_JSONL}" \
    --output-dir "${OVERFIT_DIR}" \
    --base-model "${BASE_MODEL}" \
    --initial-adapter "${BALANCED_ADAPTER}" \
    --max-length 1024 \
    --max-steps 1200 \
    --save-steps 400 \
    --logging-steps 20 \
    --learning-rate 5e-5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --seed 20260621 \
    >"${LOG_DIR}/02_train_overfit128.log" 2>&1
else
  log "overfit128 sanity adapter already exists"
fi

log "evaluating overfit128 train greedy"
mkdir -p "${OVERFIT_EVAL_DIR}"
"${PYTHON_BIN}" scripts/eval_checkpoint.py \
  --manifest "${OVERFIT_MANIFEST}" \
  --split countdown_overfit128 \
  --output-dir "${OVERFIT_EVAL_DIR}" \
  --model-name "${BASE_MODEL}" \
  --checkpoint "${OVERFIT_DIR}/final" \
  --training-mode lora \
  --batch-size 4 \
  --max-new-tokens 4096 \
  --prompt-style qwen_chat_minimal_target \
  >"${LOG_DIR}/03_eval_overfit128_train_4096.log" 2>&1

if [[ ! -f "${UNIQUE_JSONL}" ]]; then
  log "building unique-puzzle low-solution SFT data"
  "${PYTHON_BIN}" scripts/build_countdown_sft_data.py \
    --local-jsonl "${COUNTDOWN_JSONL}" \
    --exclude-manifest "${EVAL_MANIFEST}" \
    --exclude-split countdown_eval \
    --output "${UNIQUE_JSONL}" \
    --metadata-output "${UNIQUE_META}" \
    --max-scan-rows 490364 \
    --traces-per-puzzle 1 \
    --prompt-style qwen_chat_minimal_target \
    --trace-type short_success \
    --min-numbers 3 \
    --max-numbers 4 \
    --solution-cap 5 \
    --bucket-quotas ${UNIQUE_BUCKET_QUOTAS} \
    >"${LOG_DIR}/04_build_unique20k_sft_data.log" 2>&1
else
  log "unique-puzzle SFT data already exists"
fi

if [[ ! -f "${UNIQUE_DIR}/final/adapter_model.safetensors" ]]; then
  log "training unique20k continuation adapter"
  "${PYTHON_BIN}" scripts/experiments/run_countdown_sft_adaptation.py \
    --train-jsonl "${UNIQUE_JSONL}" \
    --output-dir "${UNIQUE_DIR}" \
    --base-model "${BASE_MODEL}" \
    --initial-adapter "${BALANCED_ADAPTER}" \
    --max-length 1024 \
    --max-steps "${UNIQUE_MAX_STEPS}" \
    --save-steps "${UNIQUE_SAVE_STEPS}" \
    --logging-steps 50 \
    --learning-rate "${UNIQUE_LEARNING_RATE}" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --seed 20260621 \
    >"${LOG_DIR}/05_train_unique20k.log" 2>&1
else
  log "unique20k continuation adapter already exists"
fi

log "evaluating unique20k full100 greedy"
mkdir -p "${UNIQUE_EVAL_DIR}"
"${PYTHON_BIN}" scripts/eval_checkpoint.py \
  --manifest "${EVAL_MANIFEST}" \
  --split countdown_eval \
  --output-dir "${UNIQUE_EVAL_DIR}" \
  --model-name "${BASE_MODEL}" \
  --checkpoint "${UNIQUE_DIR}/final" \
  --training-mode lora \
  --batch-size 4 \
  --max-new-tokens 4096 \
  --prompt-style qwen_chat_minimal_target \
  >"${LOG_DIR}/06_eval_unique20k_full100_4096.log" 2>&1

log "running unique20k full100 sampled audit"
mkdir -p "${UNIQUE_AUDIT_DIR}"
"${PYTHON_BIN}" scripts/experiments/audit_rollout_distribution.py \
  --manifest "${EVAL_MANIFEST}" \
  --split countdown_eval \
  --limit 100 \
  --checkpoint "${UNIQUE_DIR}/final" \
  --model-name "${BASE_MODEL}" \
  --training-mode lora \
  --prompt-style qwen_chat_minimal_target \
  --output-dir "${UNIQUE_AUDIT_DIR}" \
  --num-generations 8 \
  --batch-size 4 \
  --max-new-tokens 4096 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 20260621 \
  >"${LOG_DIR}/07_audit_unique20k_full100_g8_4096.log" 2>&1

log "Countdown final-chance run finished"
