# M1: Solver and Verifier Foundation

## Goal

Implement the exact standard 24-point foundation that all later SFT, GRPO, and reporting depend on.

## Non-Goals

- Do not train or download language models.
- Do not implement GRPO.
- Do not add Countdown-style arbitrary-target training.
- Do not copy code from LLM4Game24 or other external baselines.
- Do not broaden the answer contract beyond `<answer>...</answer>`.

## Deliverables

- `src/game24_rl/solver.py`: exact rational solver for four-number standard 24-point puzzles.
- `src/game24_rl/verifier.py`: strict verifier using AST allowlist + `Fraction`, with no Python `eval`.
- `src/game24_rl/datasets.py` and `scripts/make_splits.py`: deterministic multiset-isolated split manifest generation.
- `src/game24_rl/data_gen.py` and `scripts/build_sft_v1.py`: enough structure to generate short-success-trace SFT records after solver output exists.
- Tests in `tests/test_solver.py`, `tests/test_verifier.py`, and `tests/test_data_gen.py`.

## Required Behavior

- Enumerating all 4-number multisets from `1..13` yields `1,820` puzzles.
- The solver classifies exactly `1,362` puzzles as solvable and `458` as unsolvable.
- The verifier accepts valid tagged answers such as `<answer>((8 - 2) * (7 - 3))</answer>` for puzzle `8 2 7 3`.
- The verifier rejects missing tags, multiple answer tags, invalid syntax, unsupported AST nodes, wrong target value, wrong numbers, repeated numbers, omitted numbers, floats, names, calls, attributes, division by zero, and expressions outside the answer contract.
- Split manifests have no sorted-multiset overlap across train, validation, and test.

## Implementation Guidance

- Prefer small pure functions with explicit inputs and outputs.
- Use `fractions.Fraction` for solver and verifier arithmetic.
- Keep public functions documented with Google-style docstrings.
- Determinism matters more than cleverness for v1 data generation.
- If a verifier edge case is ambiguous, add a test and record the decision in ADR or architecture docs.

## Verification

Run at minimum:

```sh
./scripts/harness-check.sh
python -m compileall src scripts
pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py
```

Before marking this feature `passing`, also run:

```sh
ruff check .
ruff format --check .
pytest
```

## Completion Evidence

Update `harness/feature_list.json` with command results and any generated manifest artifact paths. Update `harness/session-handoff.md` with remaining risks and the next best action.
