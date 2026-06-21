"""Build a reproducible sample manifest for Jiayi-Pan/Countdown-Tasks-3to4."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


DEFAULT_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DEFAULT_SPLIT = "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-split", default="countdown_sample")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-numbers", type=int, default=3)
    parser.add_argument("--max-numbers", type=int, default=4)
    args = parser.parse_args()

    manifest = build_countdown_manifest(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        manifest_split=args.manifest_split,
        sample_size=args.sample_size,
        seed=args.seed,
        shuffle=args.shuffle,
        min_numbers=args.min_numbers,
        max_numbers=args.max_numbers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


def build_countdown_manifest(
    *,
    dataset_name: str,
    dataset_split: str,
    manifest_split: str,
    sample_size: int,
    seed: int,
    shuffle: bool,
    min_numbers: int,
    max_numbers: int,
) -> dict[str, Any]:
    """Loads Countdown and returns a Game24-compatible manifest."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on remote env.
        raise SystemExit(f"datasets is required to build Countdown manifest: {exc}")

    dataset = load_dataset(dataset_name, split=dataset_split)
    records = []
    for index, row in enumerate(dataset):
        numbers = _extract_numbers(row)
        target = _extract_target(row)
        if not min_numbers <= len(numbers) <= max_numbers:
            continue
        records.append(
            {
                "id": f"countdown-{index:07d}-{'-'.join(str(n) for n in numbers)}-t{target}",
                "key": f"{target}:{' '.join(str(number) for number in numbers)}",
                "numbers": numbers,
                "target": target,
                "dataset_index": index,
            }
        )

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(records)
    selected = records[:sample_size]
    if len(selected) < sample_size:
        raise ValueError(
            f"requested {sample_size} records but only {len(selected)} matched filters"
        )

    return {
        "version": "countdown-tasks-3to4-sample-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "seed": seed,
        "shuffle": shuffle,
        "identity": "dataset_index_numbers_target",
        "splits": {manifest_split: selected},
        "counts": {
            manifest_split: len(selected),
            "matched_before_sampling": len(records),
            "sample_size": sample_size,
        },
    }


def _extract_numbers(row: dict[str, Any]) -> list[int]:
    for key in ("nums", "numbers", "input_numbers"):
        value = row.get(key)
        if value is not None:
            return [int(number) for number in value]
    raise ValueError(f"cannot find numbers field in row keys: {sorted(row)}")


def _extract_target(row: dict[str, Any]) -> int:
    for key in ("target", "answer"):
        value = row.get(key)
        if value is not None:
            return int(value)
    raise ValueError(f"cannot find target field in row keys: {sorted(row)}")


if __name__ == "__main__":
    main()
