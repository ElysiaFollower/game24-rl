#!/usr/bin/env bash
set -euo pipefail

# Evaluate the frozen handoff1 GRPO adapter with direct greedy 4096 decoding.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
GRPO_RUN_ROOT="${GRPO_RUN_ROOT:-outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000}"
GRPO_ADAPTER="${GRPO_ADAPTER:-$GRPO_RUN_ROOT/train/final}"
MANIFEST="${MANIFEST:-data/processed/splits/standard-game24-v1.json}"
TOT_MANIFEST="${TOT_MANIFEST:-data/processed/splits/official-tot-overnight-v1.json}"
TOT_SPLIT="${TOT_SPLIT:-tot_all_1362}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_chat}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
EVAL_ROOT="${EVAL_ROOT:-$GRPO_RUN_ROOT/eval_greedy_4096}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-$GRPO_RUN_ROOT/eval_greedy_4096/by_tot_rank_from_raw.json}"

run_eval() {
  local split="$1"
  local output_dir="$EVAL_ROOT/$split"
  mkdir -p "$output_dir"
  echo "[run] eval $split -> $output_dir"
  python scripts/eval_checkpoint.py \
    --manifest "$MANIFEST" \
    --split "$split" \
    --output-dir "$output_dir" \
    --model-name "$BASE_CHECKPOINT" \
    --checkpoint "$GRPO_ADAPTER" \
    --training-mode lora \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt-style "$PROMPT_STYLE" \
    2>&1 | tee "$output_dir/eval.log"
}

run_eval validation
run_eval test

echo "[run] ToT-rank audit from raw outputs"
python scripts/audit_eval_by_tot_rank.py \
  --tot-manifest "$TOT_MANIFEST" \
  --tot-split "$TOT_SPLIT" \
  --raw-outputs "$EVAL_ROOT/validation/validation-raw-outputs.jsonl" \
  --name handoff1_grpo_validation_4096_raw \
  --raw-outputs "$EVAL_ROOT/test/test-raw-outputs.jsonl" \
  --name handoff1_grpo_test_4096_raw \
  --output "$AUDIT_OUTPUT"

echo "[done] GRPO eval artifacts: $EVAL_ROOT"
