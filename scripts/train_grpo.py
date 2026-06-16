"""Run safe GRPO dry-run and compatibility probe steps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.cli import train_grpo_main  # noqa: E402

if __name__ == "__main__":
    train_grpo_main()
