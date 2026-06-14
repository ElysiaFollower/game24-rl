# Experiment Plan

## Objective

Optimize for the strongest reliable course-visible 24-point result first. The first fallback result is a strong SFT warm start; GRPO is the frontier stage for improving beyond that checkpoint.

## Milestones

### M1: Solver and Verifier Foundation

Acceptance criteria:

- Enumerating all 4-number multisets from `1..13` yields `1,820` puzzles.
- Solver classifies `1,362` as solvable and `458` as unsolvable.
- Verifier accepts correct expressions and rejects wrong numbers, wrong values, invalid syntax, unsupported AST nodes, and division by zero.
- Verifier evaluates only the expression inside `<answer>...</answer>`.

### M2: First-Pass SFT

Training scope:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: LoRA SFT
- Task: standard 24-point only
- Data: about eight short success traces per training puzzle
- Output: `<think>short success trace</think><answer>expression</answer>`

Success gates:

- ID validation `solve_rate >= 70%`: visible fallback result
- ID validation `solve_rate >= 80%`: strong fallback result
- `format_rate >= 95%`
- `valid_expr_rate >= 80%`

If SFT v1 solve rate is below `50%`, debug the pipeline before GRPO.

### M3: GRPO Frontier Stage

Start only after M2 passes. Use the SFT checkpoint as the initial policy.

Initial GRPO goals:

- Preserve or improve SFT greedy solve rate.
- Monitor reward hacking and length collapse.
- Track `reward_mean`, `reward_std`, zero-std group rate, completion length, KL, format rate, and solve rate.

### M4: Ablations and Extensions

Candidate experiments after the fallback result:

- Increase traces per puzzle.
- Add rollback traces as an ablation.
- Add near-miss self-verification traces.
- Active-difficulty GRPO sampling.
- Countdown-style arbitrary-target OOD extension.

## Splits

Primary standard 24-point splits are by sorted multiset, never by generated trace row.

Planned evaluation surfaces:

- In-distribution validation and test splits.
- ToT hard split as a hard external evaluation.
- Unsolvable generated split for hallucination/fabrication checks.
- Countdown-style split as OOD evaluation only in the first phase.

## First-Pass SFT Data Contract

Example:

```text
<think>
(8) - (2) = 6, left: 6, 7, 3
(7) - (3) = 4, left: 6, 4
(6) * (4) = 24, left: 24
</think>
<answer>((8 - 2) * (7 - 3))</answer>
```

No rollback in SFT v1. Rollback traces are a later ablation.
