#!/usr/bin/env bash
# Create or update the Miniconda environment for local development or training.

set -euo pipefail

PROFILE="${1:-dev}"
ENV_NAME="${CONDA_ENV_NAME:-game24-rl}"

case "$PROFILE" in
  dev)
    ENV_FILE="environment.yml"
    ;;
  train)
    ENV_FILE="environment-train.yml"
    ;;
  *)
    echo "usage: $0 [dev|train]" >&2
    exit 2
    ;;
esac

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  export PATH="$HOME/miniconda3/bin:$PATH"
  # shellcheck disable=SC1091
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -x "$HOME/miniconda/bin/conda" ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  # shellcheck disable=SC1091
  . "$HOME/miniconda/etc/profile.d/conda.sh"
else
  echo "conda not found. Install Miniconda to \$HOME/miniconda3 and retry." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"

if [ "$PROFILE" = "train" ]; then
  python -m pip install --no-build-isolation -e ".[dev,train]"
else
  python -m pip install --no-build-isolation -e ".[dev]"
fi

./init.sh
