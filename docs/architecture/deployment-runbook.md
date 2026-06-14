# Deployment Runbook

This runbook is the handoff path for installing the project environment before
development or remote training.

## Goal

Make setup a single, repeatable step after the machine prerequisites are in
place. The project should be installable from `pyproject.toml` plus the conda
environment file, without rediscovering the same setup problems every time.

## Local Install

If conda is already available:

```sh
./scripts/bootstrap_conda_env.sh dev
```

For training dependencies:

```sh
./scripts/bootstrap_conda_env.sh train
```

If you only need the Python package in an existing conda environment:

```sh
python -m pip install --no-build-isolation -e '.[dev]'
```

For the training extras only:

```sh
python -m pip install --no-build-isolation -e '.[dev,train]'
```

## Remote RTXpro6000 Facts

- Machine: `rtxpro6000`
- User: `runner`
- Workdir: `/home/runner/projects/`
- Miniconda is installed at `/home/runner/miniconda3`
- Root filesystem had about `25G` free during the last check

## Remote Setup Order

1. Make sure the code is in a public remote or a clean remote copy.
2. Forward the local proxy to the remote host.
3. Export proxy variables in the remote shell.
4. Run `./scripts/bootstrap_conda_env.sh train`.
5. If it succeeds, keep the environment and move to smoke tests or training.

Example proxy flow:

```sh
ssh -N -R 127.0.0.1:7890:127.0.0.1:<local_proxy_port> RTXpro6000
```

Remote shell:

```sh
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export PATH="$HOME/miniconda3/bin:$PATH"
./scripts/bootstrap_conda_env.sh train
```

## Known Failure Modes

- `conda not found`: conda is not on `PATH`; prepend `$HOME/miniconda3/bin`.
- DNS failure for `repo.anaconda.com` or `conda.anaconda.org`: start proxy
  forwarding and export proxy variables before rerunning bootstrap.
- Long `conda` retries with no progress: treat as network blockage, not as a
  normal wait state.
- Disk pressure: the RTX host had limited free space, so do not cache large
  models or duplicate environments there without checking space first.

## Recommended Preflight

Run a fast network probe before a long install:

```sh
python - <<'PY'
import urllib.request
for url in [
    "https://repo.anaconda.com/pkgs/main/terms.json",
    "https://conda.anaconda.org/conda-forge/linux-64/repodata.json",
]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            print(url, r.status)
    except Exception as e:
        print(url, "FAILED", e)
PY
```

If that probe fails, do not wait on the bootstrap script. Fix the proxy or
cache first.

## Handoff Rule

The deployment step is complete when the environment install finishes or when
the exact blocker is recorded with a next action.
