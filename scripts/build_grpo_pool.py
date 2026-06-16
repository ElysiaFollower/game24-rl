"""Audit sampled rollout details and write a GRPO pool manifest."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.cli import build_grpo_pool_main  # noqa: E402

if __name__ == "__main__":
    build_grpo_pool_main()
