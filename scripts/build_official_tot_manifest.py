"""Build the official ToT/NLILE manifests for the overnight experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.datasets import puzzle_id, puzzle_key  # noqa: E402
from game24_rl.solver import normalize_puzzle  # noqa: E402


DEFAULT_NLILE_PATH = Path("data/raw/hf/nlile__24-game/default__train.jsonl")
DEFAULT_TOT_PATH = Path("data/raw/hf/test-time-compute__game-of-24/default__train.jsonl")
DEFAULT_OUTPUT = Path("data/processed/splits/official-tot-overnight-v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlile-path", type=Path, default=DEFAULT_NLILE_PATH)
    parser.add_argument("--tot-path", type=Path, default=DEFAULT_TOT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hard-start", type=int, default=900)
    parser.add_argument("--hard-end", type=int, default=1000)
    args = parser.parse_args()

    nlile_records = _load_nlile_records(args.nlile_path)
    tot_records = _load_tot_records(args.tot_path)
    manifest = build_manifest(
        nlile_records=nlile_records,
        tot_records=tot_records,
        hard_start=args.hard_start,
        hard_end=args.hard_end,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


def build_manifest(
    *,
    nlile_records: list[dict[str, Any]],
    tot_records: list[dict[str, Any]],
    hard_start: int,
    hard_end: int,
) -> dict[str, Any]:
    nlile_keys = {record["key"] for record in nlile_records}
    tot_keys = {record["key"] for record in tot_records}
    if nlile_keys != tot_keys:
        raise ValueError(
            "nlile/24-game and test-time-compute/game-of-24 puzzle sets differ: "
            f"nlile_only={len(nlile_keys - tot_keys)}, "
            f"tot_only={len(tot_keys - nlile_keys)}"
        )
    if len(nlile_keys) != len(nlile_records):
        raise ValueError("nlile/24-game has duplicate normalized puzzles")
    if len(tot_keys) != len(tot_records):
        raise ValueError("test-time-compute/game-of-24 has duplicate normalized puzzles")
    if not (0 <= hard_start < hard_end <= len(tot_records)):
        raise ValueError(
            f"invalid hard index range [{hard_start}, {hard_end}) for "
            f"{len(tot_records)} ToT records"
        )

    tot_all = list(tot_records)
    tot_easy = [
        record
        for record in tot_records
        if not hard_start <= int(record["tot_index"]) < hard_end
    ]
    tot_hard = [
        record
        for record in tot_records
        if hard_start <= int(record["tot_index"]) < hard_end
    ]
    return {
        "version": "official-tot-overnight-v1",
        "target": 24,
        "identity": "sorted_multiset",
        "source_datasets": {
            "nlile/24-game": {
                "path": str(DEFAULT_NLILE_PATH),
                "rows": len(nlile_records),
                "unique_puzzles": len(nlile_keys),
            },
            "test-time-compute/game-of-24": {
                "path": str(DEFAULT_TOT_PATH),
                "rows": len(tot_records),
                "unique_puzzles": len(tot_keys),
            },
        },
        "hard_subset": {
            "dataset": "test-time-compute/game-of-24",
            "index_start_inclusive_zero_based": hard_start,
            "index_end_exclusive_zero_based": hard_end,
            "rank_start_inclusive": hard_start + 1,
            "rank_end_inclusive": hard_end,
        },
        "splits": {
            "train_full_1362": [_strip_eval_only_fields(record) for record in nlile_records],
            "train_remove_900to1000_1262": [
                _strip_eval_only_fields(record) for record in tot_easy
            ],
            "tot_all_1362": tot_all,
            "tot_easy1262": tot_easy,
            "tot_hard100": tot_hard,
        },
        "counts": {
            "train_full_1362": len(nlile_records),
            "train_remove_900to1000_1262": len(tot_easy),
            "tot_all_1362": len(tot_all),
            "tot_easy1262": len(tot_easy),
            "tot_hard100": len(tot_hard),
            "overlap_unique_puzzles": len(nlile_keys & tot_keys),
            "nlile_only_unique_puzzles": len(nlile_keys - tot_keys),
            "tot_only_unique_puzzles": len(tot_keys - nlile_keys),
        },
    }


def _load_nlile_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(_read_jsonl(path)):
        numbers = list(normalize_puzzle([int(number) for number in row["numbers"]]))
        key = puzzle_key(numbers)
        records.append(
            {
                "id": f"nlile-{puzzle_id(numbers)}",
                "key": key,
                "numbers": numbers,
                "target": 24,
                "nlile_index": index,
                "solvable": bool(row.get("solvable", True)),
            }
        )
    return records


def _load_tot_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(_read_jsonl(path)):
        numbers = list(normalize_puzzle([int(part) for part in row["Puzzles"].split()]))
        key = puzzle_key(numbers)
        records.append(
            {
                "id": f"tot-{index:04d}-{puzzle_id(numbers)}",
                "key": key,
                "numbers": numbers,
                "target": 24,
                "tot_index": index,
                "tot_rank": row.get("Rank"),
                "tot_puzzles": row.get("Puzzles"),
                "tot_solved_rate": row.get("Solved rate"),
                "tot_amt_seconds": row.get("AMT (s)"),
            }
        )
    return records


def _strip_eval_only_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "key": record["key"],
        "numbers": record["numbers"],
        "target": record["target"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
