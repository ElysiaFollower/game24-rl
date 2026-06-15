# Baseline Gap Analysis

## Scope

This note compares the local SFT path with the LLM4Game24 reference baseline
after fixed single-path trace probes underperformed.

Reference clone: `/tmp/LLM4Game24`.

## Current Evidence

Validation on this repo's split and strict verifier:

| Run | Best validation solve rate | Notes |
|---|---:|---|
| v2 short-success LoRA | `42/136 = 30.88%` | Format and valid expression reached 100%. |
| v3 checked-chat LoRA | `32/136 = 23.53%` | More traces and chat prompt did not improve v2. |
| fixed trace LoRA `lr=1e-4` | `21/136 = 15.44%` | Format 100%, wrong-value dominates. |
| fixed trace LoRA `lr=3e-4` | `19/136 = 13.97%` | Higher LR did not improve short probe. |

Local recount of LLM4Game24 saved outputs:

| Baseline saved output | Strict success |
|---|---:|
| `format_v2.json` | `84/100` |
| `format_v1.json` | `63/100` |
| `short.json` | `57/100` |
| `medium.json` | `72/100` |
| `long.json` | `50/100` |

## Major Differences

### 1. Full fine-tuning vs LoRA

LLM4Game24 trains the full model with `Trainer` and
`AutoModelForCausalLM.from_pretrained(...)`; it does not use PEFT/LoRA.

This repo's SFT path has used LoRA with rank 64, alpha 128, dropout 0.05.

Why this matters: the current failures look like arithmetic/path-selection
generalization failures, not output-format failures. Full fine-tuning may be a
larger factor than trace wording for this narrow task.

Evidence status: strong code difference, not isolated yet.

Experiment: run fixed or format-v2-like data with full fine-tuning on a short
budget, using the same strict validation.

### 2. Data density per puzzle

LLM4Game24 `format_v1` and `format_v2` contain `45,353` records over `1,262`
unique train puzzles, usually `33-36` records per puzzle.

This repo:

| Dataset | Records | Unique train puzzles | Approx records per puzzle |
|---|---:|---:|---:|
| fixed trace | `1,089` | `1,089` | `1` |
| v2 short-success | `6,215` | `1,089` | up to `8` |
| v3 checked-chat | `11,706` | `1,089` | up to `32`, but only success paths |

Why this matters: fixed trace proved format learning but not path choice. The
baseline exposes many randomized search/compression views per puzzle, which may
teach state-space exploration rather than only a final success path.

Evidence status: strong data difference; baseline `format_v2 > format_v1`
suggests rollback/search context helps.

Experiment: generate repo-native search traces with `~30` records per puzzle,
with shuffled inputs and compressed search logs, while keeping strict verifier
and answer contract.

### 3. Search-tree trace vs single path

LLM4Game24 `format_v2` is a compressed DFS search tree. It preserves the
solution path and many failed branches/rollbacks. `format_v1` removes rollback
lines; both keep a final `reach 24! expression: ...` line.

Fixed trace has exactly one successful path and no negative branches.

Why this matters: fixed trace output often follows local arithmetic but ends at
wrong values. That is consistent with the model learning a step format without
learning how to choose promising operations.

Evidence status: medium-to-strong. Converted baseline format-v2 beat converted
format-v1 in earlier experiments, but both were still far below reference
baseline under our LoRA/contract setup.

Experiment: compare three data designs under the same training recipe:
single-path fixed trace, multi-solution success-only trace, and compressed
search trace.

### 4. Checkpoint selection

LLM4Game24 uses validation loss during training:

- `evaluation_strategy="steps"`
- `eval_steps=args.eval_steps`
- `load_best_model_at_end=True`
- `metric_for_best_model="loss"`
- `save_total_limit=1`

This repo mostly evaluates saved checkpoints after the fact and uses final or
manually selected checkpoints.

Why this matters: prior runs often improve slowly or non-monotonically. Best
loss checkpoint selection may prevent reporting a degraded final checkpoint.

Evidence status: weak-to-medium. It cannot explain a 15-30% vs 84% gap alone,
but should be copied for serious SFT runs.

Experiment: add a held-out train-internal eval split and save/load best model by
eval loss for long SFT experiments.

### 5. Max length and generation budget

LLM4Game24 trains with `max_len` around `1250-1400` and evaluates with
`max_new_tokens=4096`.

This repo used `max_length=512` for short/fixed traces and `max_new_tokens=256`
for validation. This is enough for fixed trace but not enough for full
baseline-style long search traces.

Evidence status: important only if we move to search traces.

Experiment: for compressed search traces, use `max_length>=1024` and
`max_new_tokens>=1024`.

### 6. Output contract

LLM4Game24 ends with `reach 24! expression: ...`; this repo uses
`<answer>...</answer>`.

Current evidence: fixed trace has `format_rate=100%`, so answer contract is not
the dominant failure for fixed trace. However, previous converted long traces
had answer-contract failures, so long search traces may need special care to
keep answer closure stable.

Experiment: keep `<answer>` but add a final trace line that explicitly mirrors
the answer, e.g. `final: <expression> = 24`, then `<answer>expression</answer>`.

## Current Best Hypotheses

1. The largest unresolved gap is full fine-tuning vs LoRA.
2. The second largest gap is data density and randomized search-trace diversity.
3. Fixed single-path trace solved format learning but did not teach path
   selection.
4. Pure learning-rate increase under LoRA is not sufficient.

## Recommended Next Experiments

1. Let the running long `lr=1e-3` LoRA fixed-trace experiment finish. It tests
   whether extreme LoRA strength and longer training rescue the single-path
   assumption.
2. If it stays low, prioritize full fine-tuning on v2 or fixed trace for a short
   controlled run.
3. In parallel or next, implement repo-native compressed search traces with
   around `30` records per puzzle and strict validation.
4. Add TensorBoard and best-checkpoint-by-eval-loss to serious SFT runs.
