# M2: SFT Training Readiness

## Goal

Turn the M1 foundation into a reproducible first-pass LoRA SFT training and
evaluation pipeline that can run on the RTXpro6000 runner, save checkpoints for
interruption recovery, and leave logs/artifacts for local analysis.

## Non-Goals

- Do not implement GRPO.
- Do not change the primary model, answer contract, verifier, or split policy.
- Do not start a long remote GPU training run until the public repository URL
  and pushed code state are available.
- Do not download model weights in local CPU-only development.
- Do not commit generated datasets, checkpoints, downloaded models, runtime
  logs, or remote-machine cache files.

## Deliverables

- `src/game24_rl/train_sft.py`: LoRA SFT entrypoint with checkpoint resume,
  periodic saves, run metadata, and logs.
- `src/game24_rl/evaluate.py` and `scripts/eval_checkpoint.py`: strict
  verifier-based checkpoint evaluation with machine-readable output artifacts.
- `configs/sft_v1.yaml`: complete enough to drive a dry run and real remote
  LoRA SFT run.
- `pyproject.toml`, `environment.yml`, `environment-train.yml`, and
  `scripts/bootstrap_conda_env.sh`: Miniconda-first local/dev/training install
  path with package console entrypoints.
- A tiny local dry-run path that exercises data loading, formatting,
  checkpoint/log directory creation, and verifier metrics without downloading
  `Qwen/Qwen2.5-1.5B-Instruct`.
- Updated tests for configuration loading, artifact schemas, and strict
  verifier metrics.

## Required Behavior

- Training can resume from an explicit checkpoint path or from the latest
  checkpoint in a run directory.
- Training writes enough metadata to recover model, data manifest, split,
  verifier version, seed, config, and resume source.
- Evaluation output declares model/checkpoint, split manifest, decoding config,
  answer contract, verifier version, and raw output artifact.
- Remote-operation docs remain aligned with the RTXpro6000 constraints:
  `remote-runner`, user `runner`, `~/projects/`, interruptible jobs, logs copied
  back for local analysis, and no new long GPU jobs after 19:00 Beijing time.
- Dry-run verification works locally without downloading large model weights.
- Local setup can be bootstrapped with `./scripts/bootstrap_conda_env.sh dev`
  for development or `./scripts/bootstrap_conda_env.sh train` for training.

## Implementation Guidance

- Keep scripts as thin wrappers over package functions.
- Use explicit JSON/JSONL artifacts rather than free-form logs for anything
  that may appear in the report.
- Prefer a small config dataclass/parser over ad hoc dictionary access.
- Add actionable errors for missing manifest, missing checkpoint, and invalid
  resume settings.
- If a training-library API detail is uncertain, isolate it behind a narrow
  function and cover the surrounding artifact/resume behavior with tests.

## Verification

Run at minimum:

```sh
./scripts/harness-check.sh
python -m compileall src scripts
pytest tests/test_solver.py tests/test_verifier.py tests/test_data_gen.py
pytest
ruff check .
ruff format --check .
```

Add focused M2 tests as implementation lands. Before any real remote run, first
run the local dry-run command and record artifact paths.

## Completion Evidence

Update `harness/feature_list.json` with command results and dry-run artifact
paths. Update `harness/session-handoff.md` with remote readiness, blockers, and
the next best action.
