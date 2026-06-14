# Remote Training Operations

This document records operational facts for long-running training and evaluation
jobs.

## RTXpro6000 Runner

- Remote access should use the `remote-runner` tool when GPU execution is
  needed.
- Host label: `RTXpro6000`.
- Remote user: `runner`.
- Work directory: `~/projects/`.
- The remote user should clone the public repository directly; do not assume
  GitHub login or private-repository access on the runner.

## Scheduling Discipline

- Beijing time is the scheduling reference.
- Long GPU jobs should be interruptible and resumable.
- Avoid starting new long GPU jobs after 19:00 Beijing time so the machine can
  be released for other users.
- After 19:00 Beijing time, prefer checkpoint sync, log collection, analysis,
  and low-cost CPU checks.

## Persistence Requirements

- Training scripts must save checkpoints often enough to survive interruption.
- Every training and evaluation run should write logs and machine-readable
  artifacts that can be copied back for local analysis.
- Reported results must reference checkpoint path, split manifest, decoding
  settings, verifier version, and raw output artifact.
- Runtime logs, downloaded models, checkpoints, and generated datasets remain
  out of git unless a small manifest or summary is intentionally committed.

## Environment

- Use Miniconda as the primary environment manager.
- Development environment: `./scripts/bootstrap_conda_env.sh dev`.
- Training environment: `./scripts/bootstrap_conda_env.sh train`.
- On `rtxpro6000`, Miniconda is already installed for `runner` at
  `/home/runner/miniconda3`; the bootstrap script auto-detects it.
- If the runner cannot reach package indexes, pre-stage a conda env or wheel
  cache before starting long jobs.
- If package-index DNS is unavailable, use a local mirror, cache, or prebuilt
  environment rather than starting a long GPU job.
- The canonical setup and proxy-forwarding steps are in
  `docs/architecture/deployment-runbook.md`.
