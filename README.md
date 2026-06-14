# Game24 RL

Owned implementation for training and evaluating a `Qwen2.5-1.5B-Instruct` solver for the 24-point game.

The project prioritizes a strong, reproducible course result:

1. Build a strict solver/verifier/evaluation foundation.
2. Train a strong SFT warm start with short success traces.
3. Use GRPO from that checkpoint to push beyond the supervised fallback result.

## Current Plan

- Primary model: `Qwen/Qwen2.5-1.5B-Instruct`
- First training task: standard 24-point puzzles only, target `24`, four numbers from `1..13`
- First SFT data: about 10k R1-wrapped short success traces
- Main answer contract: only `<answer>...</answer>` is evaluated
- Verifier: AST allowlist + `Fraction`, no Python `eval`

See:

- [CONTEXT.md](CONTEXT.md) for project language
- [docs/experiment_plan.md](docs/experiment_plan.md) for the current execution plan
- [docs/baselines.md](docs/baselines.md) for reference baselines
- [docs/architecture/deployment-runbook.md](docs/architecture/deployment-runbook.md) for environment setup and remote proxy flow
- [docs/research/](docs/research/) for the prior research reports
- [AGENTS.md](AGENTS.md) and [harness/session-handoff.md](harness/session-handoff.md) for developer handoff
- [docs/adr/](docs/adr/) for recorded decisions

## Repository Layout

```text
configs/              Training and evaluation configs
docs/                 Plans, baseline notes, and ADRs
scripts/              Thin command-line wrappers
src/game24_rl/        Package source
tests/                Unit tests for solver, verifier, and data generation
```

## Environment

This repository is Miniconda-first.

```sh
./scripts/bootstrap_conda_env.sh dev
conda activate game24-rl
```

For RTX training:

```sh
./scripts/bootstrap_conda_env.sh train
conda activate game24-rl
```

On `rtxpro6000`, conda and package-index access may need proxy forwarding.
See [docs/architecture/deployment-runbook.md](docs/architecture/deployment-runbook.md)
for the exact remote proxy flow and failure modes.

If you only want the package in an existing conda env:

```sh
python -m pip install --no-build-isolation -e '.[dev]'
```

The installed CLI entrypoints mirror the thin scripts:

```sh
game24-make-splits --output data/processed/splits/standard-game24-v1.json
game24-build-sft-v1 --manifest data/processed/splits/standard-game24-v1.json
game24-train-sft --config configs/sft_v1.yaml --dry-run
game24-eval-checkpoint --manifest data/processed/splits/standard-game24-v1.json --solver-dry-run
```

## Development Status

M1 solver/verifier/data split foundation is implemented and passing local
tests. M2 SFT readiness is active: local dry-run training/evaluation paths and
the Miniconda/package bootstrap are in place, while the first real RTX training
run is still pending a public remote clone and a working remote training
environment.
