#!/usr/bin/env bash
set -euo pipefail

# Frozen handoff1 GRPO run. Keep this aligned with
# docs/experiments/handoff1_grpo_final_plan_20260619.md.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/experiments/baseline_format_v2_full_5000_from800/final}"
MANIFEST="${MANIFEST:-data/processed/splits/standard-game24-v1.json}"
TOT_MANIFEST="${TOT_MANIFEST:-data/processed/splits/official-tot-overnight-v1.json}"
TOT_SPLIT="${TOT_SPLIT:-tot_all_1362}"
SPLIT="${SPLIT:-train}"
PROMPT_STYLE="${PROMPT_STYLE:-qwen_chat}"
RUN_ROOT="${RUN_ROOT:-outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BETA="${BETA:-0}"
SCALE_REWARDS="${SCALE_REWARDS:-none}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
REWARD_PROFILE="${REWARD_PROFILE:-closure_control_smooth}"

POOL_MANIFEST="$RUN_ROOT/pool/pool-manifest.json"
DRY_RUN_DIR="$RUN_ROOT/dry_run"
TRAIN_DIR="$RUN_ROOT/train"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

cat > "$RUN_ROOT/run-config.json" <<EOF
{
  "schema_version": "handoff1-grpo-final-run-config-v1",
  "base_checkpoint": "$BASE_CHECKPOINT",
  "manifest": "$MANIFEST",
  "tot_manifest": "$TOT_MANIFEST",
  "tot_split": "$TOT_SPLIT",
  "split": "$SPLIT",
  "prompt_style": "$PROMPT_STYLE",
  "max_new_tokens": $MAX_NEW_TOKENS,
  "num_generations": $NUM_GENERATIONS,
  "max_steps": $MAX_STEPS,
  "save_steps": $SAVE_STEPS,
  "logging_steps": $LOGGING_STEPS,
  "learning_rate": "$LEARNING_RATE",
  "beta": $BETA,
  "scale_rewards": "$SCALE_REWARDS",
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "reward_profile": "$REWARD_PROFILE",
  "mask_truncated_completions": false,
  "remove_unused_columns": false,
  "training_mode": "LoRA adapter"
}
EOF

run_step() {
  local name="$1"
  shift
  echo "[run] $name"
  echo "$*" | tee "$LOG_DIR/${name}.cmd"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

run_step 01_build_pool \
  python scripts/build_handoff1_grpo_pool.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --tot-manifest "$TOT_MANIFEST" \
    --tot-split "$TOT_SPLIT" \
    --output "$POOL_MANIFEST"

run_step 02_dry_run \
  python scripts/train_grpo.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --output-dir "$DRY_RUN_DIR" \
    --prompt-style "$PROMPT_STYLE" \
    --reward-profile "$REWARD_PROFILE" \
    --dry-run

COMMON_GRPO=(
  python scripts/train_grpo.py
  --manifest "$MANIFEST"
  --split "$SPLIT"
  --output-dir "$TRAIN_DIR"
  --prompt-style "$PROMPT_STYLE"
  --model-name-or-path "$BASE_CHECKPOINT"
  --pool-manifest "$POOL_MANIFEST"
  --max-steps "$MAX_STEPS"
  --save-steps "$SAVE_STEPS"
  --logging-steps "$LOGGING_STEPS"
  --max-completion-length "$MAX_NEW_TOKENS"
  --num-generations "$NUM_GENERATIONS"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --beta "$BETA"
  --scale-rewards "$SCALE_REWARDS"
  --reward-profile "$REWARD_PROFILE"
  --peft-mode lora
  --no-mask-truncated-completions
  --no-remove-unused-columns
)

run_step 03_compat_probe "${COMMON_GRPO[@]}" --compat-probe
run_step 04_train_grpo "${COMMON_GRPO[@]}" --train

echo "[done] GRPO final adapter: $TRAIN_DIR/final"
