# Full Fine-Tuning Search-Trace SFT Run

## Status

This is the current strong SFT result for this repository:
`110 / 136 = 80.88%` validation solve rate with the strict verifier.

The run is an owned recipe: repo data, repo split, repo strict verifier, and the
repository `<answer>...</answer>` reported contract.

## Model

- Base model family: `Qwen/Qwen2.5-1.5B-Instruct`.
- Training type: full fine-tuning.
- Precision: `bf16`.
- Optimizer: `adamw_torch`.
- Weight decay: `0.01`.
- Learning rate: `5e-5`.
- Schedule: warmup ratio `0.03`, then cosine decay.
- Effective batch size: `8` (`per_device_train_batch_size=1`,
  `gradient_accumulation_steps=8`).
- Max sequence length: `1024`.

## Data

- Split manifest: `data/processed/splits/standard-game24-v1.json`.
- Training split: `train`.
- Evaluation split: `validation`.
- Training records:
  `data/processed/experiments/game24-baseline-format-v2-qwen-answer-train.jsonl`.
- Records: `35324`.
- Unique train multisets: `1011`.
- Search/rollback records: `32355`.
- Validation multiset overlap: `0`.
- Prompt style: `qwen_chat`.
- Answer contract for reported metrics: exactly one expression inside
  `<answer>...</answer>`.
- Verifier: repository strict AST + `Fraction` verifier.

## Training Chain

The final result came from staged full fine-tuning:

1. Short full fine-tuning probe to `400` steps.
2. Continue from the `400`-step local full checkpoint to `800` steps.
3. Continue from the `800`-step local full checkpoint to `5000` steps.

The decisive run used:

- Initialization:
  `outputs/experiments/baseline_format_v2_full_continue800/checkpoint-800`.
- Run directory:
  `outputs/experiments/baseline_format_v2_full_5000_from800`.
- Max steps: `5000`.
- Save steps: `500`.
- Runtime: about `1:28:22`.
- Final train loss: `0.06988`.
- Final Trainer eval loss: `0.06184`.
- Best saved checkpoint by Trainer eval loss:
  `outputs/experiments/baseline_format_v2_full_5000_from800/checkpoint-4500`
  with eval loss `0.0617639`.

Launch command:

```sh
python scripts/experiments/run_rollback_sft_experiment.py \
  --mode train \
  --dataset data/processed/experiments/game24-baseline-format-v2-qwen-answer-train.jsonl \
  --run-dir outputs/experiments/baseline_format_v2_full_5000_from800 \
  --training-mode full \
  --prompt-style qwen_chat \
  --max-steps 5000 \
  --save-steps 500 \
  --max-length 1024 \
  --max-new-tokens 1024 \
  --eval-batch-size 4 \
  --model-name outputs/experiments/baseline_format_v2_full_continue800/checkpoint-800
```

## Strict Validation Result

Evaluation command:

```sh
python scripts/experiments/run_rollback_sft_experiment.py \
  --mode eval \
  --dataset data/processed/experiments/game24-baseline-format-v2-qwen-answer-train.jsonl \
  --run-dir outputs/experiments/baseline_format_v2_full_5000_from800 \
  --training-mode full \
  --prompt-style qwen_chat \
  --max-new-tokens 1024 \
  --eval-batch-size 4 \
  --model-name outputs/experiments/baseline_format_v2_full_5000_from800/final
```

Evaluation artifact:
`outputs/experiments/baseline_format_v2_full_5000_from800/eval/summary.json`.

| Checkpoint | Solve rate | Format rate | Valid expression rate |
|---|---:|---:|---:|
| `checkpoint-4500` | `110 / 136 = 80.88%` | `80.88%` | `80.88%` |
| `final` | `110 / 136 = 80.88%` | `80.88%` | `80.88%` |

## Interpretation

- The earlier `400`-step result was undertrained for this recipe.
- Longer full fine-tuning raised strict validation solve rate from `37.50%` to
  `80.88%`.
- Format, valid-expression, and solve rates are identical because every
  format-valid output in this evaluation also passed strict solving.
- The current SFT model is strong enough to serve as the project fallback and as
  a credible warm start for the next optimization stage.

## Training Curve

Trainer scalar history is stored in `trainer_state.json` for the saved
checkpoint. Export plots with:

```sh
python scripts/export_training_metrics.py \
  --run-dir outputs/experiments/baseline_format_v2_full_5000_from800 \
  --state outputs/experiments/baseline_format_v2_full_5000_from800/checkpoint-4500/trainer_state.json \
  --output-dir outputs/experiments/baseline_format_v2_full_5000_from800/metrics
```

Outputs:

- `training_metrics.csv`
- `loss.svg`
- `learning_rate.svg`

Note: because `save_total_limit=1`, the retained `trainer_state.json` covers the
latest retained checkpoint (`checkpoint-4500`). The final step metrics are still
visible in the training log:
`outputs/experiments/baseline_format_v2_full_5000_from800/logs/train-20260616-baseline-v2-full-5000-from800.log`.
