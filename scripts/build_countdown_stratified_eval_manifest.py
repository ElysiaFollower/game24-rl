"""Build a DFS-solution-count-stratified Countdown eval manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.countdown_solver import count_countdown_solutions  # noqa: E402

DEFAULT_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DEFAULT_SPLIT = "train"
DEFAULT_BUCKETS = ("1", "2", "3", "4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-split", default="countdown_eval")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--max-scan-rows", type=int, default=200_000)
    parser.add_argument("--solution-cap", type=int, default=5)
    parser.add_argument("--min-numbers", type=int, default=3)
    parser.add_argument("--max-numbers", type=int, default=4)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        manifest_split=args.manifest_split,
        sample_size=args.sample_size,
        buckets=tuple(args.buckets),
        max_scan_rows=args.max_scan_rows,
        solution_cap=args.solution_cap,
        min_numbers=args.min_numbers,
        max_numbers=args.max_numbers,
        allow_partial=args.allow_partial,
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
    dataset_name: str,
    dataset_split: str,
    manifest_split: str,
    sample_size: int,
    buckets: tuple[str, ...],
    max_scan_rows: int,
    solution_cap: int,
    min_numbers: int,
    max_numbers: int,
    allow_partial: bool,
) -> dict[str, Any]:
    """Streams Countdown rows and selects a balanced solvable eval set."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if max_scan_rows <= 0:
        raise ValueError("max_scan_rows must be positive")
    if solution_cap < 2:
        raise ValueError("solution_cap must be at least 2")
    if not buckets:
        raise ValueError("at least one bucket is required")

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on remote env.
        raise SystemExit(
            f"datasets is required to build Countdown manifest: {exc}"
        ) from exc

    quotas = _bucket_quotas(sample_size, buckets)
    selected_by_bucket: dict[str, list[dict[str, Any]]] = {
        bucket: [] for bucket in buckets
    }
    observed_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    scanned_rows = 0

    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    for index, row in enumerate(dataset):
        if scanned_rows >= max_scan_rows:
            break
        scanned_rows += 1

        try:
            numbers = _extract_numbers(row)
            target = _extract_target(row)
        except ValueError:
            skipped_counts["bad_schema"] += 1
            continue
        if not min_numbers <= len(numbers) <= max_numbers:
            skipped_counts["number_count_out_of_range"] += 1
            continue

        count_result = count_countdown_solutions(
            numbers,
            target,
            cap=solution_cap,
        )
        bucket = _bucket_for_count(count_result.count, solution_cap=solution_cap)
        observed_counts[bucket] += 1
        if count_result.count == 0:
            continue
        if bucket not in selected_by_bucket:
            continue
        if len(selected_by_bucket[bucket]) >= quotas[bucket]:
            continue
        selected_by_bucket[bucket].append(
            {
                "id": (
                    f"countdown-{index:07d}-"
                    f"{'-'.join(str(n) for n in numbers)}-t{target}"
                ),
                "key": f"{target}:{' '.join(str(number) for number in numbers)}",
                "numbers": numbers,
                "target": target,
                "dataset_index": index,
                "solution_count": count_result.count,
                "solution_count_bucket": bucket,
                "solution_count_capped": count_result.capped,
                "solvable": True,
            }
        )
        if _quotas_filled(selected_by_bucket, quotas):
            break

    selected = [
        record
        for bucket in buckets
        for record in selected_by_bucket[bucket]
    ]
    missing = {
        bucket: max(0, quotas[bucket] - len(selected_by_bucket[bucket]))
        for bucket in buckets
    }
    if any(missing.values()) and not allow_partial:
        summary = {
            "scanned_rows": scanned_rows,
            "observed_solution_count_buckets": dict(sorted(observed_counts.items())),
            "selected_by_bucket": {
                bucket: len(records)
                for bucket, records in selected_by_bucket.items()
            },
            "requested_by_bucket": quotas,
            "missing_by_bucket": missing,
            "skipped_counts": dict(sorted(skipped_counts.items())),
        }
        raise SystemExit(
            "insufficient solvable Countdown rows for requested stratified sample:\n"
            + json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        )

    return {
        "version": "countdown-tasks-3to4-stratified-eval-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "identity": "dataset_index_numbers_target",
        "stratification": {
            "method": "exact_dfs_unique_expression_count",
            "buckets": list(buckets),
            "solution_cap": solution_cap,
            "zero_solution_rows_excluded": True,
            "bucket_note": (
                f"{solution_cap}_plus means at least {solution_cap} unique "
                "verifier-style expressions"
            ),
        },
        "splits": {manifest_split: selected},
        "counts": {
            manifest_split: len(selected),
            "sample_size_requested": sample_size,
            "scanned_rows": scanned_rows,
            "requested_by_bucket": quotas,
            "selected_by_bucket": {
                bucket: len(records)
                for bucket, records in selected_by_bucket.items()
            },
            "observed_solution_count_buckets": dict(sorted(observed_counts.items())),
            "missing_by_bucket": missing,
            "skipped_counts": dict(sorted(skipped_counts.items())),
        },
    }


def _bucket_quotas(sample_size: int, buckets: tuple[str, ...]) -> dict[str, int]:
    base = sample_size // len(buckets)
    remainder = sample_size % len(buckets)
    return {
        bucket: base + int(index < remainder)
        for index, bucket in enumerate(buckets)
    }


def _bucket_for_count(count: int, *, solution_cap: int) -> str:
    if count <= 0:
        return "0"
    if count >= solution_cap:
        return f"{solution_cap}_plus"
    return str(count)


def _quotas_filled(
    selected_by_bucket: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
) -> bool:
    return all(
        len(selected_by_bucket[bucket]) >= quota
        for bucket, quota in quotas.items()
    )


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
