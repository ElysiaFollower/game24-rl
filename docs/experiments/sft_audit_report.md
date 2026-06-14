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
