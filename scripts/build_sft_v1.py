"""Build the first-pass short-success-trace SFT dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.cli import build_sft_v1_main  # noqa: E402

if __name__ == "__main__":
    build_sft_v1_main()
