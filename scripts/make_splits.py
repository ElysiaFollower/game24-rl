"""Create reproducible train, validation, test, and OOD split manifests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.cli import make_splits_main  # noqa: E402

if __name__ == "__main__":
    make_splits_main()
