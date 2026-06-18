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

## Checkpoint Trajectory

The available strict validation points form this sparse trajectory:

| Step | Run directory | Checkpoint | Solve rate | Format rate | Valid expression rate | Main remaining failure |
|---:|---|---|---:|---:|---:|---|
| `400` | `baseline_format_v2_full` | `checkpoint-400` | `51 / 136 = 37.50%` | `38.97%` | `38.24%` | `83` answer-contract failures |
| `800` | `baseline_format_v2_full_continue800` | `checkpoint-800` | `70 / 136 = 51.47%` | `54.41%` | `54.41%` | `62` answer-contract failures |
| `4500` | `baseline_format_v2_full_5000_from800` | `checkpoint-4500` | `110 / 136 = 80.88%` | `80.88%` | `80.88%` | `26` answer-contract failures |
| `5000` | `baseline_format_v2_full_5000_from800` | `final` | `110 / 136 = 80.88%` | `80.88%` | `80.88%` | `26` answer-contract failures |

Local analysis artifact:
`outputs/experiments/baseline_format_v2_full_5000_from800/analysis/checkpoint_trajectory_and_failures.json`.

## Interpretation

- The earlier `400`-step result was undertrained for this recipe.
- Longer full fine-tuning raised strict validation solve rate from `37.50%` to
  `80.88%`.
- Format, valid-expression, and solve rates are identical because every
  format-valid output in this evaluation also passed strict solving.
- The equal `checkpoint-4500` and `final` metrics are not only aggregate-level:
  all `136 / 136` raw greedy validation outputs are byte-identical between the
  two checkpoints. The plausible explanation is that cosine-decayed learning
  rate was already very low by step `4500` (`1.30e-6` in the retained
  Trainer history), so the final `500` steps did not cross any greedy decoding
  decision boundary on this validation set.
- At `4500` and `final`, the remaining `26` failures all fail the answer
  contract and contain no `<answer>` block. Case-study samples show long
  rollback/search continuations truncated by the `1024` new-token budget before
  emitting `</think><answer>...`.
- The current SFT model is strong enough to serve as the project fallback and as
  a credible warm start for the next optimization stage.

## Follow-Up Idea: Canonical DFS Trace

The rollback/search trace recipe is useful as a strong baseline, but it may not
be the best teacher format. A possible follow-up is to replace random rollback
search with a more canonical DFS trace: fixed pair/operator order, explicit
state deduplication, shorter successful paths, and an immediate answer closure
once a solution is found.

Potential benefit: this may teach more systematic search and reduce the current
failure mode where the model keeps emitting long rollback traces until the token
budget is exhausted.

Potential risk: a deterministic DFS teacher may make the model imitate a rigid
search procedure, reducing the flexible search behavior that random traces may
encourage. Because this is speculative and not required to reproduce the strong
SFT baseline, it is not part of the current official-data rerun.

## Generation Budget Interpretation

The `max_new_tokens=1024` cap is an important evaluation choice, not a neutral
implementation detail.

For the current result, the cap makes the remaining `26` failures easy to
interpret mechanically: the model keeps producing rollback/search steps and is
truncated before it emits a complete `<answer>...</answer>` block. This does not
prove that those puzzles are impossible for the model with a longer budget; it
may be that some validation puzzles require longer search traces under this
teacher format.

At the same time, keeping a bounded generation budget is intentional. The target
behavior is not exhaustive brute-force enumeration until a solution appears; the
project wants a model that reaches a valid solution promptly under a declared
decoding budget. Under this interpretation, truncation failures are legitimate
failures of search control and answer closure, even if a larger budget might
recover some cases.

Future reports should therefore state `max_new_tokens` together with accuracy.
If budget sensitivity becomes a research question, evaluate the same checkpoint
under multiple budgets, for example `512`, `1024`, and `2048`, and report both
solve rate and failure mix.

## Training Curve

This historical run did not write TensorBoard events because the experimental
script still used `report_to=["none"]` at the time. Future runs should keep the
default `--report-to tensorboard`, which writes TensorBoard event files under the
run logging directory.

For this run, Trainer scalar history is stored in `trainer_state.json` for the
saved checkpoint. Export plots with:

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

Observed curve summary:

- Retained scalar records: `450` log rows from step `10` to `4500`.
- Window mean loss fell from about `0.0803` over steps `10..500` to about
  `0.0635` over steps `4001..4500`.
- Learning rate followed warmup plus cosine decay: about `4.80e-5` mean over
  steps `501..1000`, `7.79e-6` over `3501..4000`, and `2.96e-6` over
  `4001..4500`.

Note: because `save_total_limit=1`, the retained `trainer_state.json` covers the
latest retained checkpoint (`checkpoint-4500`). The final step metrics are still
visible in the training log:
`outputs/experiments/baseline_format_v2_full_5000_from800/logs/train-20260616-baseline-v2-full-5000-from800.log`.

## Observability Follow-Up

This run did not retain enough intermediate checkpoints for a dense accuracy
curve. Future long SFT runs that may need checkpoint-wise validation should set
`--save-total-limit` larger than `1` and keep TensorBoard enabled. The rollback
experiment script now exposes `--save-total-limit` for this purpose while
preserving the old default.
