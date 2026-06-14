#!/usr/bin/env bash
# Validate the repository on a remote training machine before long GPU jobs.
# Prefer running this inside the Miniconda env from bootstrap_conda_env.sh.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_MODE="${REMOTE_INSTALL_MODE:-venv}"
SKIP_INSTALL="${REMOTE_SKIP_INSTALL:-0}"
PIP_INSTALL_FLAGS=()

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "== identity =="
whoami
pwd
date

echo "== python =="
"$PYTHON_BIN" --version
if [ "$INSTALL_MODE" = "venv" ]; then
  if "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python --version
    python -m pip install --upgrade pip
  else
    echo "venv creation failed; set REMOTE_INSTALL_MODE=system-user if apt python3-venv is unavailable." >&2
    exit 1
  fi
elif [ "$INSTALL_MODE" = "system-user" ]; then
  PIP_INSTALL_FLAGS=(--user)
  if "$PYTHON_BIN" -m pip install --help | grep -q -- --break-system-packages; then
    PIP_INSTALL_FLAGS+=(--break-system-packages)
  fi
  python() {
    "$PYTHON_BIN" "$@"
  }
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "unknown REMOTE_INSTALL_MODE: $INSTALL_MODE" >&2
  exit 1
fi

echo "== install package =="
if [ "$SKIP_INSTALL" = "1" ]; then
  echo "Skipping install step because REMOTE_SKIP_INSTALL=1"
else
  python -m pip install "${PIP_INSTALL_FLAGS[@]}" -e ".[dev]"
fi

echo "== local validation =="
./init.sh
python -m compileall src scripts
pytest
if command -v ruff >/dev/null 2>&1; then
  ruff check .
  ruff format --check .
else
  echo "ruff not found; skipping lint checks on remote machine"
fi

echo "== dry-run artifacts =="
python scripts/train_sft.py --config configs/sft_v1.yaml --dry-run
python scripts/eval_checkpoint.py \
  --manifest data/processed/splits/standard-game24-v1.json \
  --solver-dry-run \
  --output-dir outputs/eval/sft_v1_dryrun \
  --split validation \
  --limit 16
