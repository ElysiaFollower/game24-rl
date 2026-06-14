# Evaluation Foundation

This document records stable design facts for the first implementation slice.

## Puzzle Identity

- A standard puzzle is a sorted multiset of four integers from `1..13`.
- Split membership is decided by the sorted multiset, not by generated trace rows.
- The M1 count target is `1,820` total multisets, `1,362` solvable, and `458` unsolvable.

## Solver

- The exact solver should use rational arithmetic throughout.
- The solver may choose any internal search representation, but outputs must be deterministic enough for reproducible data generation.
- Short success traces are the default SFT v1 trace source; rollback traces are a later ablation.

## Verifier

- The verifier extracts exactly one expression from `<answer>...</answer>`.
- The verifier evaluates with an AST allowlist and `Fraction`.
- Allowed arithmetic is binary `+`, `-`, `*`, `/` over the provided numbers.
- The expression must use exactly the input multiset and evaluate exactly to target `24`.
- Python `eval`, regex-only validation, floats, imports, calls, attributes, names, and unsupported AST nodes are invalid.

## Evaluation

- Reported scores must state model, checkpoint, split manifest, decoding settings, answer contract, and verifier version.
- In-distribution validation/test and OOD evaluation must be reported separately.
- LLM4Game24 remains a reference baseline; its output format is not this repository's main answer contract.
