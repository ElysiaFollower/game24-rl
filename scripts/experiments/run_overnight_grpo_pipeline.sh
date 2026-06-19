#!/usr/bin/env bash
set -euo pipefail

# This script only orchestrates independently reproducible experiment scripts.
# Override paths/settings through environment variables instead of editing it.

RUN_ROOT="${RUN_ROOT:-outputs/official_nlile_strong_sft_20260618/grpo_pipeline_$(date +%Y%m%d_%H%M%S)}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-outputs/official_nlile_strong_sft_20260618/sft/SFT-full-nlile-single-5000-plus1500-from-final/final}"
GAME24_MANIFEST="${GAME24_MANIFEST:-data/processed/splits/official-tot-overnight-v1.json}"
GAME24_BASELINE_REPORT="${GAME24_BASELINE_REPORT:-outputs/official_nlile_strong_sft_20260618/eval/SFT-full-nlile-single-5000-plus1500-4096/tot_all_1362-eval-report.json}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"

EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"

GAME24_GRPO_STEPS="${GAME24_GRPO_STEPS:-100}"
GAME24_GRPO_MAX_COMPLETION_LENGTH="${GAME24_GRPO_MAX_COMPLETION_LENGTH:-2048}"
GAME24_ROLLOUT_MAX_NEW_TOKENS="${GAME24_ROLLOUT_MAX_NEW_TOKENS:-2048}"
GAME24_ROLLOUT_NUM_GENERATIONS="${GAME24_ROLLOUT_NUM_GENERATIONS:-8}"

COUNTDOWN_SAMPLE_SIZE="${COUNTDOWN_SAMPLE_SIZE:-200}"
COUNTDOWN_TRAIN_SAMPLE_SIZE="${COUNTDOWN_TRAIN_SAMPLE_SIZE:-400}"
COUNTDOWN_GRPO_STEPS="${COUNTDOWN_GRPO_STEPS:-100}"
COUNTDOWN_GRPO_MAX_COMPLETION_LENGTH="${COUNTDOWN_GRPO_MAX_COMPLETION_LENGTH:-2048}"
COUNTDOWN_ROLLOUT_MAX_NEW_TOKENS="${COUNTDOWN_ROLLOUT_MAX_NEW_TOKENS:-2048}"
COUNTDOWN_ROLLOUT_NUM_GENERATIONS="${COUNTDOWN_ROLLOUT_NUM_GENERATIONS:-8}"
COUNTDOWN_EVAL_SEED="${COUNTDOWN_EVAL_SEED:-20260618}"
COUNTDOWN_TRAIN_SEED="${COUNTDOWN_TRAIN_SEED:-20260619}"

mkdir -p "$RUN_ROOT/logs"
{
  echo "run_root=$RUN_ROOT"
  echo "sft_checkpoint=$SFT_CHECKPOINT"
  echo "game24_manifest=$GAME24_MANIFEST"
  echo "game24_baseline_report=$GAME24_BASELINE_REPORT"
  echo "eval_max_new_tokens=$EVAL_MAX_NEW_TOKENS"
  echo "game24_grpo_steps=$GAME24_GRPO_STEPS"
  echo "countdown_sample_size=$COUNTDOWN_SAMPLE_SIZE"
  echo "countdown_train_sample_size=$COUNTDOWN_TRAIN_SAMPLE_SIZE"
  echo "countdown_grpo_steps=$COUNTDOWN_GRPO_STEPS"
} | tee "$RUN_ROOT/run-settings.txt"

FAILURE_REPORT_ARGS=()
if [[ -f "$GAME24_BASELINE_REPORT" ]]; then
  FAILURE_REPORT_ARGS=(--failure-report "$GAME24_BASELINE_REPORT")
else
  echo "[warn] baseline report not found, Game24 GRPO rollout will use the whole split: $GAME24_BASELINE_REPORT" | tee -a "$RUN_ROOT/run-settings.txt"
fi

python scripts/experiments/run_game24_grpo.py \
  --run-root "$RUN_ROOT/game24_grpo" \
  --manifest "$GAME24_MANIFEST" \
  --split tot_all_1362 \
  --sft-checkpoint "$SFT_CHECKPOINT" \
  --model-name "$MODEL_NAME" \
  "${FAILURE_REPORT_ARGS[@]}" \
  --rollout-num-generations "$GAME24_ROLLOUT_NUM_GENERATIONS" \
  --rollout-max-new-tokens "$GAME24_ROLLOUT_MAX_NEW_TOKENS" \
  --grpo-max-steps "$GAME24_GRPO_STEPS" \
  --grpo-max-completion-length "$GAME24_GRPO_MAX_COMPLETION_LENGTH"

python scripts/experiments/eval_game24_model.py \
  --manifest "$GAME24_MANIFEST" \
  --split tot_all_1362 \
  --output-dir "$RUN_ROOT/eval/game24_grpo_tot_all_4096" \
  --model-name "$SFT_CHECKPOINT" \
  --checkpoint "$RUN_ROOT/game24_grpo/train/final" \
  --training-mode lora \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --prompt-style qwen_chat

python scripts/experiments/eval_game24_model.py \
  --manifest "$GAME24_MANIFEST" \
  --split train_full_1362 \
  --output-dir "$RUN_ROOT/eval/game24_grpo_train_full_4096" \
  --model-name "$SFT_CHECKPOINT" \
  --checkpoint "$RUN_ROOT/game24_grpo/train/final" \
  --training-mode lora \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --prompt-style qwen_chat \
  --no-group-summary

python scripts/experiments/eval_countdown_sample.py \
  --output-dir "$RUN_ROOT/eval/countdown_sft_baseline_sample" \
  --model-name "$MODEL_NAME" \
  --checkpoint "$SFT_CHECKPOINT" \
  --training-mode full \
  --sample-size "$COUNTDOWN_SAMPLE_SIZE" \
  --seed "$COUNTDOWN_EVAL_SEED" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS"

python scripts/experiments/run_countdown_grpo.py \
  --run-root "$RUN_ROOT/countdown_grpo" \
  --start-checkpoint "$SFT_CHECKPOINT" \
  --model-name "$MODEL_NAME" \
  --sample-size "$COUNTDOWN_TRAIN_SAMPLE_SIZE" \
  --seed "$COUNTDOWN_TRAIN_SEED" \
  --rollout-num-generations "$COUNTDOWN_ROLLOUT_NUM_GENERATIONS" \
  --rollout-max-new-tokens "$COUNTDOWN_ROLLOUT_MAX_NEW_TOKENS" \
  --grpo-max-steps "$COUNTDOWN_GRPO_STEPS" \
  --grpo-max-completion-length "$COUNTDOWN_GRPO_MAX_COMPLETION_LENGTH"

python scripts/experiments/eval_countdown_sample.py \
  --output-dir "$RUN_ROOT/eval/countdown_grpo_sample" \
  --model-name "$SFT_CHECKPOINT" \
  --checkpoint "$RUN_ROOT/countdown_grpo/train/final" \
  --training-mode lora \
  --sample-size "$COUNTDOWN_SAMPLE_SIZE" \
  --seed "$COUNTDOWN_EVAL_SEED" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS"

python scripts/experiments/eval_game24_model.py \
  --manifest "$GAME24_MANIFEST" \
  --split tot_all_1362 \
  --output-dir "$RUN_ROOT/eval/countdown_grpo_game24_retention_4096" \
  --model-name "$SFT_CHECKPOINT" \
  --checkpoint "$RUN_ROOT/countdown_grpo/train/final" \
  --training-mode lora \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  --prompt-style qwen_chat

echo "[done] overnight GRPO pipeline completed: $RUN_ROOT"
