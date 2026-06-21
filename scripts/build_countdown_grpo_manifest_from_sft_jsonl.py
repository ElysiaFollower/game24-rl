"""Build a balanced Countdown GRPO prompt manifest from SFT JSONL records."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-split", default="countdown_grpo_train")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument(
        "--bucket-quotas",
        nargs="+",
        required=True,
        help="Per solution-count bucket quotas, e.g. 1=128 2=128 3=128 4=128.",
    )
    args = parser.parse_args()

    quotas = _parse_bucket_quotas(args.bucket_quotas)
    manifest = build_manifest(
        sft_jsonl=args.sft_jsonl,
        manifest_split=args.manifest_split,
        bucket_quotas=quotas,
        seed=args.seed,
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
    sft_jsonl: Path,
    manifest_split: str,
    bucket_quotas: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """Builds a unique-prompt manifest from solver-verified SFT records."""

    by_key: dict[str, dict[str, Any]] = {}
    records_read = 0
    with sft_jsonl.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records_read += 1
            row = json.loads(line)
            bucket = str(row["solution_count_bucket"])
            if bucket not in bucket_quotas:
                continue
            numbers = [int(number) for number in row["numbers"]]
            target = int(row["target"])
            dataset_index = int(row["dataset_index"])
            key = f"{target}:{' '.join(str(number) for number in numbers)}"
            by_key.setdefault(
                key,
                {
                    "id": (
                        f"countdown-grpo-{dataset_index:07d}-"
                        f"{'-'.join(str(number) for number in numbers)}-t{target}"
                    ),
                    "key": key,
                    "numbers": numbers,
                    "target": target,
                    "dataset_index": dataset_index,
                    "solution_count": int(row["solution_count"]),
                    "solution_count_bucket": bucket,
                    "solution_count_capped": bool(row["solution_count_capped"]),
                    "solvable": True,
                },
            )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in by_key.values():
        grouped[str(record["solution_count_bucket"])].append(record)

    rng = random.Random(seed)
    selected_by_bucket: dict[str, list[dict[str, Any]]] = {}
    missing_by_bucket: dict[str, int] = {}
    for bucket, quota in bucket_quotas.items():
        candidates = sorted(grouped.get(bucket, []), key=lambda item: item["id"])
        rng.shuffle(candidates)
        selected = candidates[:quota]
        selected_by_bucket[bucket] = sorted(selected, key=lambda item: item["id"])
        missing_by_bucket[bucket] = max(0, quota - len(selected))

    if any(missing_by_bucket.values()):
        raise SystemExit(
            "insufficient records for requested bucket quotas:\n"
            + json.dumps(
                {
                    "requested_by_bucket": bucket_quotas,
                    "available_by_bucket": {
                        bucket: len(grouped.get(bucket, []))
                        for bucket in bucket_quotas
                    },
                    "missing_by_bucket": missing_by_bucket,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    selected_records = [
        record
        for bucket in bucket_quotas
        for record in selected_by_bucket[bucket]
    ]
    return {
        "version": "countdown-grpo-from-sft-jsonl-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_sft_jsonl": str(sft_jsonl),
        "seed": seed,
        "identity": "target_ordered_numbers",
        "splits": {manifest_split: selected_records},
        "counts": {
            manifest_split: len(selected_records),
            "records_read": records_read,
            "unique_prompt_records": len(by_key),
            "requested_by_bucket": bucket_quotas,
            "selected_by_bucket": {
                bucket: len(records)
                for bucket, records in selected_by_bucket.items()
            },
            "available_by_bucket": {
                bucket: len(grouped.get(bucket, []))
                for bucket in bucket_quotas
            },
        },
    }


def _parse_bucket_quotas(items: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"bucket quota must be BUCKET=COUNT, got {item!r}")
        bucket, raw_count = item.split("=", 1)
        bucket = bucket.strip()
        count = int(raw_count)
        if not bucket:
            raise ValueError("bucket name must be non-empty")
        if count <= 0:
            raise ValueError("bucket quota must be positive")
        quotas[bucket] = count
    if not quotas:
        raise ValueError("at least one bucket quota is required")
    return quotas


if __name__ == "__main__":
    main()
