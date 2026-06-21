"""Audit puzzle overlap between downloaded HF Game24 datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.datasets import puzzle_key  # noqa: E402
from game24_rl.solver import normalize_puzzle  # noqa: E402


DEFAULT_NLILE_PATH = Path("data/raw/hf/nlile__24-game/default__train.jsonl")
DEFAULT_TOT_PATH = Path("data/raw/hf/test-time-compute__game-of-24/default__train.jsonl")
DEFAULT_OUTPUT_PATH = Path("outputs/audits/hf_dataset_overlap_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit overlap between nlile/24-game and test-time-compute/game-of-24."
    )
    parser.add_argument("--nlile-path", type=Path, default=DEFAULT_NLILE_PATH)
    parser.add_argument("--tot-path", type=Path, default=DEFAULT_TOT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--hard-start",
        type=int,
        default=900,
        help="Zero-based inclusive start index for the hard subset.",
    )
    parser.add_argument(
        "--hard-end",
        type=int,
        default=1000,
        help="Zero-based exclusive end index for the hard subset.",
    )
    args = parser.parse_args()

    nlile = _load_records(args.nlile_path, dataset="nlile")
    tot = _load_records(args.tot_path, dataset="tot")
    summary = build_overlap_summary(
        nlile=nlile,
        tot=tot,
        hard_start=args.hard_start,
        hard_end=args.hard_end,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _print_summary(summary, args.output)


def build_overlap_summary(
    *,
    nlile: list[dict[str, Any]],
    tot: list[dict[str, Any]],
    hard_start: int,
    hard_end: int,
) -> dict[str, Any]:
    nlile_key_to_records = _records_by_key(nlile)
    tot_key_to_records = _records_by_key(tot)
    nlile_keys = set(nlile_key_to_records)
    tot_keys = set(tot_key_to_records)
    overlap_keys = nlile_keys & tot_keys
    nlile_only_keys = nlile_keys - tot_keys
    tot_only_keys = tot_keys - nlile_keys

    # Keep a direct O(n^2) cross-check because the datasets are tiny and this
    # makes the audit robust against accidental key-map mistakes.
    pairwise_matches = []
    for nlile_record in nlile:
        for tot_record in tot:
            if nlile_record["key"] == tot_record["key"]:
                pairwise_matches.append(
                    {
                        "key": nlile_record["key"],
                        "nlile_index": nlile_record["index"],
                        "tot_index": tot_record["index"],
                        "tot_rank": tot_record["source"].get("Rank"),
                    }
                )

    hard_tot = [
        record for record in tot if hard_start <= record["index"] < hard_end
    ]
    hard_keys = {record["key"] for record in hard_tot}
    hard_overlap_keys = hard_keys & nlile_keys

    return {
        "datasets": {
            "nlile/24-game": {
                "rows": len(nlile),
                "unique_puzzles": len(nlile_keys),
                "duplicate_keys": _duplicate_counts(nlile),
            },
            "test-time-compute/game-of-24": {
                "rows": len(tot),
                "unique_puzzles": len(tot_keys),
                "duplicate_keys": _duplicate_counts(tot),
            },
        },
        "overlap": {
            "unique_puzzle_count": len(overlap_keys),
            "nlile_overlap_rows": sum(
                len(nlile_key_to_records[key]) for key in overlap_keys
            ),
            "tot_overlap_rows": sum(len(tot_key_to_records[key]) for key in overlap_keys),
            "pairwise_match_count": len(pairwise_matches),
            "nlile_only_unique_count": len(nlile_only_keys),
            "tot_only_unique_count": len(tot_only_keys),
            "nlile_only_examples": _example_records(nlile_key_to_records, nlile_only_keys),
            "tot_only_examples": _example_records(tot_key_to_records, tot_only_keys),
        },
        "hard_subset": {
            "definition": {
                "dataset": "test-time-compute/game-of-24",
                "index_start_inclusive_zero_based": hard_start,
                "index_end_exclusive_zero_based": hard_end,
            },
            "rows": len(hard_tot),
            "unique_puzzles": len(hard_keys),
            "overlap_with_nlile_unique_count": len(hard_overlap_keys),
            "non_overlap_with_nlile_unique_count": len(hard_keys - nlile_keys),
            "non_overlap_examples": _example_records(
                tot_key_to_records,
                hard_keys - nlile_keys,
            ),
            "first_records": [
                _compact_record(record) for record in hard_tot[:5]
            ],
            "last_records": [
                _compact_record(record) for record in hard_tot[-5:]
            ],
        },
        "pairwise_match_examples": pairwise_matches[:20],
    }


def _load_records(path: Path, *, dataset: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        source = json.loads(line)
        numbers = _extract_numbers(source, dataset=dataset)
        normalized = normalize_puzzle(numbers)
        records.append(
            {
                "index": index,
                "numbers": list(normalized),
                "key": puzzle_key(normalized),
                "source": source,
            }
        )
    return records


def _extract_numbers(source: dict[str, Any], *, dataset: str) -> list[int]:
    if dataset == "nlile":
        numbers = source.get("numbers")
        if not isinstance(numbers, list):
            raise ValueError(f"nlile row has invalid numbers: {source!r}")
        return [int(number) for number in numbers]
    if dataset == "tot":
        puzzle = source.get("Puzzles")
        if not isinstance(puzzle, str):
            raise ValueError(f"ToT row has invalid Puzzles field: {source!r}")
        return [int(part) for part in puzzle.split()]
    raise ValueError(f"unknown dataset: {dataset}")


def _records_by_key(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["key"], []).append(record)
    return grouped


def _duplicate_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["key"] for record in records)
    return {key: count for key, count in counts.items() if count > 1}


def _example_records(
    records_by_key: dict[str, list[dict[str, Any]]],
    keys: set[str],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    examples = []
    for key in sorted(keys)[:limit]:
        examples.append(_compact_record(records_by_key[key][0]))
    return examples


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": record["index"],
        "key": record["key"],
        "numbers": record["numbers"],
        "rank": record["source"].get("Rank"),
        "puzzles": record["source"].get("Puzzles"),
    }


def _print_summary(summary: dict[str, Any], output: Path) -> None:
    nlile = summary["datasets"]["nlile/24-game"]
    tot = summary["datasets"]["test-time-compute/game-of-24"]
    overlap = summary["overlap"]
    hard = summary["hard_subset"]
    print("nlile/24-game rows:", nlile["rows"])
    print("nlile/24-game unique puzzles:", nlile["unique_puzzles"])
    print("test-time-compute/game-of-24 rows:", tot["rows"])
    print("test-time-compute/game-of-24 unique puzzles:", tot["unique_puzzles"])
    print("overlap unique puzzles:", overlap["unique_puzzle_count"])
    print("nlile-only unique puzzles:", overlap["nlile_only_unique_count"])
    print("tot-only unique puzzles:", overlap["tot_only_unique_count"])
    print("pairwise match count:", overlap["pairwise_match_count"])
    print("hard subset rows:", hard["rows"])
    print("hard subset unique puzzles:", hard["unique_puzzles"])
    print("hard subset overlap with nlile:", hard["overlap_with_nlile_unique_count"])
    print("hard subset non-overlap with nlile:", hard["non_overlap_with_nlile_unique_count"])
    print("output:", output)


if __name__ == "__main__":
    main()
