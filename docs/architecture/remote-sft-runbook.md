# Remote SFT Runbook

This runbook is the operational path for M2 on `rtxpro6000`.

## Official Path

1. Push the current local repository to a public Git remote.
2. Use `remote-runner` to clone that public repository into
   `/home/runner/projects/`.
3. Create the Miniconda training environment:

   ```sh
   ./scripts/bootstrap_conda_env.sh train
   conda activate game24-rl
   ```

   On `rtxpro6000`, Miniconda is installed at `/home/runner/miniconda3`,
   and the bootstrap script auto-detects that path.
   If conda channels are unreachable, see the deployment runbook for the
   proxy-forwarding flow before rerunning bootstrap.

4. Run `scripts/remote_readiness.sh` before any long GPU job.
5. Start SFT with `scripts/train_sft.py --config configs/sft_v1.yaml
   --auto-resume`.
6. Pull back logs, `run_metadata.json`, evaluation reports, and checkpoint
   summaries for local analysis.

## Temporary Path When Git Remote Is Missing

If the local repository has no public remote yet, a tarball upload through
`remote-runner file put` may be used only for dry-run and environment
validation. Do not treat tarball-uploaded code as the official training source
for reportable scores. When packaging from macOS, exclude AppleDouble metadata
such as `._*`; those files can make remote harness checks and compileall scan
invalid pseudo-source files.

## Environment Caveat

The current runner image may not have outbound package-index access. If
`./scripts/bootstrap_conda_env.sh train` or `python -m pip install -e
".[dev,train]"` fails on build dependencies, use a prebuilt conda environment,
pre-stage wheels/cache on the remote host, or fix package-index access before
starting a long job. `REMOTE_SKIP_INSTALL=1` allows the runbook to proceed with
existing remote dependencies when only dry run and environment checks are
needed.

During the first `rtxpro6000` bootstrap, direct DNS/package-index access to
`repo.anaconda.com` failed. The Miniconda installer was fetched through a local
HTTP mirror and installed successfully, but the full training dependency set
still needs a working package path, cache, or prebuilt environment.

## Resource Discipline

- Beijing time is the scheduling reference.
- Before 07:00 Beijing time, unattended remote setup, dry-run validation, and
  bounded training are acceptable.
- Avoid starting new long GPU jobs after 19:00 Beijing time.
- Real training must be resumable with `--auto-resume` or an explicit
  `--resume-from-checkpoint`.
- Preserve logs and artifacts under `outputs/`, `runs/`, or `checkpoints/`, but
  keep those paths out of git.

## Minimum Remote Checks

```sh
./scripts/remote_readiness.sh
```

The command validates package installation, harness checks, compileall, pytest,
ruff, SFT dry-run, and solver-based evaluation dry-run without loading the large
model.
