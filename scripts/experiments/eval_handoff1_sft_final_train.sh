#!/usr/bin/env bash
set -euo pipefail

# Fill the missing handoff1 SFT-final train-split eval artifacts.
# Existing validation/test raw outputs are reused for the final ToT-rank audit.

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
MANIFEST="${MANIFEST:-data/processed/splits/standard-game24-v1.json}"
TOT_MANIFEST="${TOT_MANIFEST:-data/processed/splits/official-tot-overnight-v1.json}"
SPLIT="${SPLIT:-train}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_chat}"
TRAINING_MODE="${TRAINING_MODE:-full}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
RUN_DIR="${RUN_DIR:-outputs/experiments/baseline_format_v2_full_5000_from800/eval}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-outputs/audits/handoff1_sft_final_train_val_test_by_tot_rank_from_raw.json}"
FORCE="${FORCE:-0}"

VAL_1024_RAW="${VAL_1024_RAW:-outputs/experiments/baseline_format_v2_full_5000_from800/eval/final/validation-raw-outputs.jsonl}"
TEST_1024_RAW="${TEST_1024_RAW:-/root/autodl-tmp/projects/sft-direct-long/sft_final_test_greedy_1024/test-raw-outputs.jsonl}"
VAL_4096_RAW="${VAL_4096_RAW:-/root/autodl-tmp/projects/sft-direct-long/sft_final_validation_greedy_4096/validation-raw-outputs.jsonl}"
TEST_4096_RAW="${TEST_4096_RAW:-/root/autodl-tmp/projects/sft-direct-long/sft_final_test_greedy_4096/test-raw-outputs.jsonl}"

run_eval() {
  local max_new_tokens="$1"
  local output_dir="$RUN_DIR/final_${SPLIT}_${max_new_tokens}"
  local report="$output_dir/${SPLIT}-eval-report.json"
  local raw="$output_dir/${SPLIT}-raw-outputs.jsonl"
  local log="$output_dir/eval.log"

  mkdir -p "$output_dir"
  if [[ "$FORCE" != "1" && -s "$report" && -s "$raw" ]]; then
    echo "[skip] ${SPLIT} ${max_new_tokens}: existing artifacts found at $output_dir"
    return
  fi

  echo "[run] ${SPLIT} ${max_new_tokens}: $output_dir"
  python scripts/eval_checkpoint.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --output-dir "$output_dir" \
    --model-name "$MODEL_NAME" \
    --checkpoint "$MODEL_CHECKPOINT" \
    --training-mode "$TRAINING_MODE" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-new-tokens "$max_new_tokens" \
    --prompt-style "$PROMPT_STYLE" \
    2>&1 | tee "$log"

  test -s "$report"
  test -s "$raw"
}

run_audit() {
  local train_1024_raw="$RUN_DIR/final_${SPLIT}_1024/${SPLIT}-raw-outputs.jsonl"
  local train_4096_raw="$RUN_DIR/final_${SPLIT}_4096/${SPLIT}-raw-outputs.jsonl"

  for path in "$train_1024_raw" "$VAL_1024_RAW" "$TEST_1024_RAW" "$train_4096_raw" "$VAL_4096_RAW" "$TEST_4096_RAW"; do
    if [[ ! -s "$path" ]]; then
      echo "[error] required raw output is missing: $path" >&2
      exit 1
    fi
  done

  echo "[run] ToT-rank audit from raw outputs: $AUDIT_OUTPUT"
  python scripts/audit_eval_by_tot_rank.py \
    --tot-manifest "$TOT_MANIFEST" \
    --raw-outputs "$train_1024_raw" \
    --name sft_final_train_1024_raw \
    --raw-outputs "$VAL_1024_RAW" \
    --name sft_final_validation_1024_raw \
    --raw-outputs "$TEST_1024_RAW" \
    --name sft_final_test_1024_raw \
    --raw-outputs "$train_4096_raw" \
    --name sft_final_train_4096_raw \
    --raw-outputs "$VAL_4096_RAW" \
    --name sft_final_validation_4096_raw \
    --raw-outputs "$TEST_4096_RAW" \
    --name sft_final_test_4096_raw \
    --output "$AUDIT_OUTPUT"
}

cat <<EOF
[settings]
MODEL_CHECKPOINT=$MODEL_CHECKPOINT
MANIFEST=$MANIFEST
SPLIT=$SPLIT
RUN_DIR=$RUN_DIR
AUDIT_OUTPUT=$AUDIT_OUTPUT
EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE
FORCE=$FORCE
EOF

run_eval 1024
run_eval 4096
run_audit

echo "[done] handoff1 SFT-final train eval completed"
