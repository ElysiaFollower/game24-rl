# SFT Audit Report

## Status

`M2-sft-audit-and-repair` audit completed enough to launch the next SFT run.
The next run is a focused repair attempt, not a claim that SFT is already fixed.

## Trigger

SFT v2 remained far below the success gate after fixing the prompt/completion
separator and stale SFT JSONL cache:

| Checkpoint | Validation solve rate | Format rate | Valid expression rate |
|---|---:|---:|---:|
| `checkpoint-500` | `21 / 136` = `15.44%` | `99.26%` | `83.09%` |
| `checkpoint-1500` | `29 / 136` = `21.32%` | `100.00%` | `100.00%` |
| `final` | `42 / 136` = `30.88%` | `100.00%` | `100.00%` |

Final report: `outputs/eval/sft_v2_fixed_prompt_final_validation_20260615-010152/validation-eval-report.json` on AutoDL.

## Evaluation Audit

Reviewed `src/game24_rl/evaluate.py`, `src/game24_rl/cli.py`, and
`scripts/eval_checkpoint.py`.

Findings:

- Strict scoring uses `verify_answer(output, puzzle=record["numbers"], target=...)`.
  `solve_rate` is exactly `valid / total`; no aggregate scoring bug was found.
- `valid_expr_rate` intentionally counts `wrong_value` as syntactically valid,
  number-correct expressions. This explains `valid_expr_rate=100%` while
  `solve_rate` remains low.
- Adapter loading uses `PeftModel.from_pretrained(base_model, checkpoint)` and
  generation slices off the prompt tokens before decoding.
- Raw-output and report artifacts record model, checkpoint, split manifest,
  decoding, verifier version, answer contract, and raw-output path.
- Repair: evaluation now supports and records `generation_prompt_style`, so a
  chat-prompt-trained SFT run can be evaluated under the same prompt contract.
- Repair: generation now uses `torch.inference_mode()`.

Conclusion: the low score is not explained by an obvious verifier/reporting bug.
The dominant failure mode is genuine model output with correct format and numbers
but wrong arithmetic target.

## Data Audit

Current v2 training data:

- Path: `data/processed/sft/game24-sft-v1-train.jsonl`
- Records: `6215`
- Split overlap with validation: `0`
- Format: plain prompt plus 3-line success trace and `<answer>...</answer>`.

Case-study artifact:
`outputs/diagnostics/sft_case_study/summary.json`.

Summary from `checkpoint-1500` validation:

- `29` ok, `107` wrong_value.
- `format_ok=136/136`, `valid_expr=136/136`.
- Failure examples are structurally valid and use exactly the input numbers, but
  end at values such as `21`, `12`, `23`, `36`, or `22`.
- Nearest training examples often differ by one card and have similar-looking
  arithmetic templates, suggesting local pattern copying rather than robust
  target search.

Representative failure:

- Puzzle `[5, 5, 5, 9]`: model output `((5 - 9) + (5 * 5)) = 21`; solver has
  trivial valid solutions such as `((5 + 5) + (5 + 9))`.

## Training Audit

Reviewed `src/game24_rl/train_sft.py` and focused tests.

Findings:

- TRL SFT uses prompt/completion records with `completion_only_loss=True`.
- Checkpoint save/resume plumbing exists and was previously smoke-tested.
- Cached SFT JSONL regeneration is content-aware.
- Confirmed bug/tech debt: `SftDataConfig.trace_type` existed in config and run
  metadata but was not used by data generation. This made trace-style experiments
  non-reproducible from config alone.
- Repair: `trace_type` and `prompt_style` now flow through config, CLI, data
  generation, dry-run metadata, and evaluation artifacts.

## Baseline Reference

Reference repo inspected locally at `/tmp/LLM4Game24` from
`https://github.com/LiaoMengqi/LLM4Game24`.

Relevant differences:

- Baseline trains/evaluates Qwen with explicit Qwen chat framing:
  system/user/assistant boundaries.
- Baseline manually masks prompt labels and trains only assistant output.
- Its reported strong format-v2 data uses about `45k` records, not this repo's
  `6215` records.
- Format-v2 keeps a compact search-log style, including rollback traces in many
  records, and ends with an explicit target statement like `reach 24! expression:`.
- This repo keeps its stricter `<answer>...</answer>` contract and AST/Fraction
  verifier; Python `eval` is not used.

Interpretation: our first-pass data is likely too weak as a search teacher. It
shows only one clean success path with no explicit final target check beyond the
last arithmetic line, so the model learns the output shape but often stops at a
nearby non-24 value.

## Fixes Implemented

- Added `prompt_style` support:
  - `plain` keeps the existing prompt.
  - `qwen_chat` uses Qwen-style system/user/assistant boundaries while preserving
    the `<answer>...</answer>` answer contract.
- Added `trace_type` support:
  - `short_success` keeps existing behavior.
  - `checked_success` appends an explicit final check line inside `<think>`.
- Added `configs/sft_v3_checked_chat.yaml`:
  - same model: `Qwen/Qwen2.5-1.5B-Instruct`.
  - same split manifest and verifier.
  - `traces_per_puzzle: 32`, generating `11706` strict-verifier-valid records.
  - `trace_type: checked_success`.
  - `prompt_style: qwen_chat`.
  - run name: `qwen25_15b_lora_sft_v3_checked_chat`.
- Added focused tests covering trace/prompt config propagation and eval prompt
  style reporting.
- Formatted SFT diagnostic scripts used by this audit path.

## Verification

Passed locally:

```sh
python -m compileall src scripts
pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py tests/test_training_pipeline.py
ruff check src scripts tests
ruff format --check src scripts tests configs
python scripts/train_sft.py --config configs/sft_v3_checked_chat.yaml --dry-run
```

Manual data check:

- `data/processed/sft/game24-sft-v3-checked-chat-train.jsonl`
- records: `11706`
- strict verifier valid completions: `11706 / 11706`
- prompt tail ends with `<|im_start|>assistant\n`
- first completion includes `Check: the final expression ... = 24.`

`python scripts/audit_sft_dataset.py ...` could not run locally because this
local environment does not have `transformers`; the replacement manual audit used
repo-local JSONL reading plus strict verifier.

## Re-Train Plan

Start a new AutoDL tmux run with:

```sh
python scripts/train_sft.py --config configs/sft_v3_checked_chat.yaml
```

Evaluate the resulting checkpoints with matching prompt style:

```sh
python scripts/eval_checkpoint.py \
  --manifest data/processed/splits/standard-game24-v1.json \
  --split validation \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint outputs/sft_v1/qwen25_15b_lora_sft_v3_checked_chat/checkpoint-500 \
  --output-dir outputs/eval/sft_v3_checked_chat_ckpt500_validation \
  --prompt-style qwen_chat
```

## Residual Risk

This repair addresses confirmed config/eval reproducibility bugs and strengthens
the SFT teacher signal. It may still underperform if the decisive baseline factor
is rollback/search traces rather than chat framing, target-check statements, or
record count. If v3 remains low, the next evidence-based step is to generate a
repo-native rollback/search-trace dataset while preserving the current verifier
and answer contract.

## 2026-06-15 Update: Fixed State-Trace Experiment

After v3 and converted baseline-format experiments remained below the expected
baseline order of magnitude, the current working interpretation changed from
"likely hidden training pipeline bug" to "likely teacher/data design bottleneck,
with no core pipeline bug found yet."

Evidence behind this interpretation:

- One-sample and 16-sample probes used the same `train_sft.py`/TRL/LoRA
  save-load/eval path and overfit successfully, so there is no current evidence
  for a fatal completion mask, checkpoint, or eval slicing bug in the main path.
- Converted baseline `format_v2` data was stronger than converted `format_v1`
  under this repo's prompt and answer contract, showing that explicit search or
  state information helps.
- Long rollback/search traces also introduced many answer-contract failures,
  suggesting that they can disrupt final `<answer>` closure for this setup.
- Original short-success traces kept format stable but often produced
  correct-number wrong-value answers, suggesting that the model learned output
  shape and local arithmetic templates more than reliable state progression.

Decision: run the next SFT exploration with a deterministic single-solution
state-transition trace:

```text
<think>
<s0> 5 5 5 9
<s1> 5 + 5 = 10 | left: 10 5 9
<s2> 5 + 9 = 14 | left: 10 14
<s3> 10 + 14 = 24 | left: 24
</think>
<answer>((5 + 5) + (5 + 9))</answer>
```

Assumption: for a small model on a narrow task, a low-entropy task-specific
grammar may be easier to learn than broad natural-language search. The fixed
trace keeps useful intermediate state supervision while deferring rollback and
multi-candidate search until the model demonstrates basic state-update and
answer-closure competence.

Deferred alternatives: GRPO, verifier relaxation, answer-contract changes,
multi-candidate traces, long rollback traces, full finetuning, and aggressive
learning-rate changes. These remain possible later, but fixed trace is the next
smallest experiment that can distinguish "needs clearer state supervision" from
"still has a deeper training/data issue."

## 2026-06-15 Update: Baseline-Format v2 Full Finetune Checkpoint

This run is the current best reproducible SFT backup result in this repository:
`51 / 136 = 37.50%` validation solve rate. It is not yet acceptable as the
target result, but it is a useful checkpoint because it combines the strongest
teacher data seen so far with full finetuning instead of LoRA.

Method:

- Script: `scripts/experiments/run_rollback_sft_experiment.py`.
- Mode: `--mode train`, followed by `--mode eval`; dataset was prebuilt, so the
  expensive rollback-data builder was not rerun.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Training mode: `--training-mode full`, using Transformers `Trainer` with
  prompt masking and `optim=adamw_torch`.
- Dataset:
  `data/processed/experiments/game24-baseline-format-v2-qwen-answer-train.jsonl`.
- Dataset summary:
  - records: `35324`
  - unique train multisets: `1011`
  - rollback records: `32355`
  - validation overlap: `0`
  - prompt style: `qwen_chat`
- Main training parameters: `max_steps=400`, `save_steps=200`,
  `learning_rate=5e-5`, `max_length=1024`, `max_new_tokens=1024`,
  `eval_batch_size=4`.
- AutoDL tmux session: `game24-baseline-v2-full`.
- Run dir: `outputs/experiments/baseline_format_v2_full`.
- Eval summary:
  `outputs/experiments/baseline_format_v2_full/eval/summary.json`.

Validation result:

| Checkpoint | Solve rate | Format rate | Valid expression rate |
|---|---:|---:|---:|
| `checkpoint-400` | `51 / 136` = `37.50%` | `38.97%` | `38.24%` |
| `final` | `51 / 136` = `37.50%` | `38.97%` | `38.24%` |

Interpretation:

- This is better than the previous SFT v2 final result
  (`42 / 136 = 30.88%`), but still far below the reference baseline target.
- The low `format_rate` and `valid_expr_rate` show that the dominant remaining
  failure is not only arithmetic search. Long search/rollback traces still
  frequently fail to close or satisfy the repository answer contract.
- The result supports continuing baseline-aligned work, but the next iteration
  should inspect raw outputs before changing the objective: likely candidates
  are prompt/format closure, generation length, data formatting, and differences
  from the reference baseline's training recipe.

## 2026-06-16 Update: Strong Full Fine-Tuning Result

Longer full fine-tuning confirmed that the `400`-step result was undertrained for
the search-trace SFT recipe. Continuing from the local full checkpoint chain to
`5000` steps produced a strict validation solve rate of `110 / 136 = 80.88%`.

Independent run record:
`docs/experiments/sft_full_finetune_search_trace_20260616.md`.

Evaluation artifact:
`outputs/experiments/baseline_format_v2_full_5000_from800/eval/summary.json`.
