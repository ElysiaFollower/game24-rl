"""Run or dry-run the first-pass LoRA SFT pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.cli import train_sft_main  # noqa: E402

if __name__ == "__main__":
    train_sft_main()
