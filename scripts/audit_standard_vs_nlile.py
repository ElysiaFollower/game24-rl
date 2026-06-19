"""Audit exact puzzle overlap between standard-game24-v1, nlile/24-game, and SFT data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_STANDARD = Path("data/processed/splits/standard-game24-v1.json")
DEFAULT_NLILE = Path("data/raw/hf/nlile__24-game/default__train.jsonl")
DEFAULT_SFT = Path(
    "data/processed/experiments/game24-baseline-format-v2-qwen-answer-train.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--nlile", type=Path, default=DEFAULT_NLILE)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = audit(
        standard_path=args.standard,
        nlile_path=args.nlile,
        sft_jsonl_path=args.sft_jsonl,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")


def audit(
    *,
    standard_path: Path,
    nlile_path: Path,
    sft_jsonl_path: Path,
) -> dict[str, Any]:
    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    standard_splits = {
        split: [normalize_key(record["numbers"]) for record in records]
        for split, records in standard["splits"].items()
    }
    standard_solvable_keys = [
        key for split in ("train", "validation", "test") for key in standard_splits[split]
    ]
    nlile_rows = read_jsonl(nlile_path)
    nlile_keys = [normalize_key(row["numbers"]) for row in nlile_rows]

    result: dict[str, Any] = {
        "standard_path": str(standard_path),
        "nlile_path": str(nlile_path),
        "standard_counts": standard.get("counts", {}),
        "standard_solvable_unique": len(set(standard_solvable_keys)),
        "nlile_rows": len(nlile_rows),
        "nlile_unique": len(set(nlile_keys)),
        "standard_duplicate_keys": duplicate_counts(standard_solvable_keys),
        "nlile_duplicate_keys": duplicate_counts(nlile_keys),
        "standard_minus_nlile": sorted(set(standard_solvable_keys) - set(nlile_keys)),
        "nlile_minus_standard": sorted(set(nlile_keys) - set(standard_solvable_keys)),
        "split_counts": {split: len(keys) for split, keys in standard_splits.items()},
    }
    result["same_unique_puzzle_set"] = (
        not result["standard_minus_nlile"] and not result["nlile_minus_standard"]
    )

    if sft_jsonl_path.exists():
        sft_rows = read_jsonl(sft_jsonl_path)
        sft_keys = [normalize_key(row["numbers"]) for row in sft_rows]
        train_keys = set(standard_splits["train"])
        validation_keys = set(standard_splits["validation"])
        test_keys = set(standard_splits["test"])
        sft_unique = set(sft_keys)
        result["sft_jsonl_path"] = str(sft_jsonl_path)
        result["sft_rows"] = len(sft_rows)
        result["sft_unique_puzzles"] = len(sft_unique)
        result["sft_records_per_puzzle_min"] = min(Counter(sft_keys).values())
        result["sft_records_per_puzzle_max"] = max(Counter(sft_keys).values())
        result["sft_minus_standard_train"] = sorted(sft_unique - train_keys)
        result["standard_train_minus_sft"] = sorted(train_keys - sft_unique)
        result["sft_overlap_validation"] = sorted(sft_unique & validation_keys)
        result["sft_overlap_test"] = sorted(sft_unique & test_keys)
        result["sft_matches_standard_train_only"] = (
            not result["sft_minus_standard_train"]
            and not result["standard_train_minus_sft"]
            and not result["sft_overlap_validation"]
            and not result["sft_overlap_test"]
        )
    else:
        result["sft_jsonl_path"] = str(sft_jsonl_path)
        result["sft_missing"] = True

    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_key(numbers: list[int]) -> str:
    return " ".join(str(number) for number in sorted(int(number) for number in numbers))


def duplicate_counts(keys: list[str]) -> dict[str, int]:
    return {key: count for key, count in Counter(keys).items() if count > 1}


if __name__ == "__main__":
    main()
