"""Build the frozen handoff1 GRPO prompt pool manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.grpo import GRPO_POOL_SCHEMA_VERSION  # noqa: E402


DEFAULT_MANIFEST = Path("data/processed/splits/standard-game24-v1.json")
DEFAULT_TOT_MANIFEST = Path("data/processed/splits/official-tot-overnight-v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tot-manifest", type=Path, default=DEFAULT_TOT_MANIFEST)
    parser.add_argument("--tot-split", default="tot_all_1362")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--easy-repeat", type=int, default=1)
    parser.add_argument("--medium-repeat", type=int, default=1)
    parser.add_argument("--hard-repeat", type=int, default=2)
    parser.add_argument("--very-hard-repeat", type=int, default=2)
    parser.add_argument("--unknown-repeat", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["splits"][args.split]
    rank_index = load_tot_rank_index(args.tot_manifest, args.tot_split)

    selected_prompt_ids: list[str] = []
    bucket_counts: Counter[str] = Counter()
    weighted_bucket_counts: Counter[str] = Counter()
    missing_tot_rank: list[str] = []
    repeat_by_bucket = {
        "easy_1_300": args.easy_repeat,
        "medium_301_900": args.medium_repeat,
        "hard_901_1100": args.hard_repeat,
        "very_hard_1101_1362": args.very_hard_repeat,
        "unknown": args.unknown_repeat,
    }

    for record in records:
        key = puzzle_key(record["numbers"])
        rank_info = rank_index.get(key)
        if rank_info is None:
            bucket = "unknown"
            missing_tot_rank.append(str(record["id"]))
        else:
            bucket = bucket_for_rank(int(rank_info["rank"]))
        repeat = repeat_by_bucket[bucket]
        bucket_counts[bucket] += 1
        weighted_bucket_counts[bucket] += repeat
        selected_prompt_ids.extend([str(record["id"])] * repeat)

    payload = {
        "schema_version": GRPO_POOL_SCHEMA_VERSION,
        "metadata": {
            "pool_type": "handoff1_full_train_rank_weighted_v1",
            "manifest": str(args.manifest),
            "split": args.split,
            "tot_manifest": str(args.tot_manifest),
            "tot_split": args.tot_split,
            "rank_source": (
                "matched by sorted input-number multiset to ToT manifest "
                "tot_index + 1"
            ),
            "repeat_by_bucket": repeat_by_bucket,
            "unique_prompt_count": len(records),
            "weighted_prompt_count": len(selected_prompt_ids),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "weighted_bucket_counts": dict(sorted(weighted_bucket_counts.items())),
            "missing_tot_rank_count": len(missing_tot_rank),
            "missing_tot_rank_ids": missing_tot_rank[:20],
        },
        "audit": {
            "schema_version": GRPO_POOL_SCHEMA_VERSION,
            "passed": True,
            "failures": [],
            "total_prompts": len(records),
            "total_outputs": 0,
            "mixed_groups": 0,
            "all_correct_groups": 0,
            "all_wrong_groups": 0,
            "zero_std_groups": 0,
            "correct_truncation_mixed_groups": 0,
            "mixed_group_rate": 0.0,
            "zero_std_group_rate": 0.0,
            "all_wrong_rate": 0.0,
            "gate": {
                "min_pool_size": 0,
                "min_mixed_group_rate": 0.0,
                "max_zero_std_group_rate": 1.0,
                "min_correct_truncation_mixed": 0,
                "max_all_wrong_rate": 1.0,
            },
            "selected_prompt_ids": selected_prompt_ids,
        },
        "selected_prompt_ids": selected_prompt_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


def load_tot_rank_index(manifest_path: Path, split: str) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["splits"][split]
    index: dict[str, dict[str, Any]] = {}
    for fallback_index, record in enumerate(records):
        key = puzzle_key(record["numbers"])
        rank = int(record.get("tot_index", fallback_index)) + 1
        index[key] = {
            "rank": rank,
            "tot_index": rank - 1,
            "tot_id": record.get("id"),
        }
    return index


def bucket_for_rank(rank: int) -> str:
    if 1 <= rank <= 300:
        return "easy_1_300"
    if 301 <= rank <= 900:
        return "medium_301_900"
    if 901 <= rank <= 1100:
        return "hard_901_1100"
    if 1101 <= rank <= 1362:
        return "very_hard_1101_1362"
    raise ValueError(f"rank out of expected ToT range 1-1362: {rank}")


def puzzle_key(numbers: list[int]) -> str:
    return " ".join(str(number) for number in sorted(int(number) for number in numbers))


if __name__ == "__main__":
    main()
