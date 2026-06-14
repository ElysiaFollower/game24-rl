"""Dataset manifests and split helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from game24_rl.solver import (
    DEFAULT_TARGET,
    STANDARD_MAX_NUMBER,
    enumerate_standard_puzzles,
    is_solvable,
)

DEFAULT_SPLIT_SEED = 20260613
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VALIDATION_FRACTION = 0.1
MANIFEST_VERSION = "standard-game24-v1"


def puzzle_key(numbers: tuple[int, ...] | list[int]) -> str:
    """Returns the stable sorted-multiset key for a puzzle."""

    return " ".join(str(number) for number in sorted(numbers))


def puzzle_id(numbers: tuple[int, ...] | list[int]) -> str:
    """Returns a filesystem- and JSON-friendly puzzle id."""

    return "-".join(str(number) for number in sorted(numbers))


def build_split_manifest(
    seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    max_number: int = STANDARD_MAX_NUMBER,
    target: int = DEFAULT_TARGET,
) -> dict[str, Any]:
    """Builds a deterministic multiset-isolated split manifest.

    The train/validation/test splits contain solvable puzzles. Unsolvable
    standard puzzles are kept in a separate list for hallucination checks.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1")

    solvable_records: list[dict[str, Any]] = []
    unsolvable_records: list[dict[str, Any]] = []
    for puzzle in enumerate_standard_puzzles(max_number=max_number):
        record = _puzzle_record(puzzle, target=target)
        if is_solvable(puzzle, target=target):
            solvable_records.append(record | {"solvable": True})
        else:
            unsolvable_records.append(record | {"solvable": False})

    ordered = sorted(
        solvable_records, key=lambda record: _stable_split_key(record, seed)
    )
    train_count = int(len(ordered) * train_fraction)
    validation_count = int(len(ordered) * validation_fraction)
    train = ordered[:train_count]
    validation = ordered[train_count : train_count + validation_count]
    test = ordered[train_count + validation_count :]

    manifest = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "target": target,
        "max_number": max_number,
        "identity": "sorted_multiset",
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "unsolvable": sorted(
            unsolvable_records, key=lambda record: tuple(record["numbers"])
        ),
        "counts": {
            "total": len(solvable_records) + len(unsolvable_records),
            "solvable": len(solvable_records),
            "unsolvable": len(unsolvable_records),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    }
    validate_no_split_overlap(manifest)
    return manifest


def validate_no_split_overlap(manifest: dict[str, Any]) -> None:
    """Raises if train, validation, and test share any puzzle identity."""

    seen: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for record in manifest["splits"][split_name]:
            key = record["key"]
            if key in seen:
                raise ValueError(
                    f"puzzle {key!r} appears in both {seen[key]} and {split_name}"
                )
            seen[key] = split_name


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Writes a split manifest as pretty JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Reads a split manifest JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _puzzle_record(puzzle: tuple[int, ...], target: int) -> dict[str, Any]:
    return {
        "id": puzzle_id(puzzle),
        "key": puzzle_key(puzzle),
        "numbers": list(puzzle),
        "target": target,
    }


def _stable_split_key(record: dict[str, Any], seed: int) -> str:
    payload = f"{seed}:{record['key']}".encode()
    return hashlib.sha256(payload).hexdigest()
