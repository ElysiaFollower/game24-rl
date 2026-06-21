"""Build solver-trace SFT JSONL for Countdown target-puzzle adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.countdown_solver import (  # noqa: E402
    count_countdown_solutions,
    solve_countdown_puzzle,
)
from game24_rl.data_gen import format_completion, format_prompt  # noqa: E402
from game24_rl.verifier import VERIFIER_VERSION, verify_answer  # noqa: E402

DEFAULT_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"
DEFAULT_SPLIT = "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--local-jsonl",
        type=Path,
        help="Optional local JSONL cache of Countdown rows; avoids HF streaming.",
    )
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--max-scan-rows", type=int, default=200_000)
    parser.add_argument("--traces-per-puzzle", type=int, default=1)
    parser.add_argument("--prompt-style", default="qwen_chat_target")
    parser.add_argument("--trace-type", default="checked_success")
    parser.add_argument("--min-numbers", type=int, default=3)
    parser.add_argument("--max-numbers", type=int, default=4)
    parser.add_argument("--solution-cap", type=int, default=5)
    parser.add_argument(
        "--include-buckets",
        nargs="+",
        help=(
            "Optional solution-count buckets to include, e.g. 1 2 3 4 5_plus. "
            "When omitted, all solvable buckets are included."
        ),
    )
    parser.add_argument(
        "--bucket-quotas",
        nargs="+",
        help=(
            "Optional per-bucket record quotas, e.g. 1=5000 2=5000 3=5000 "
            "4=5000. When set, --sample-size is ignored and scanning stops "
            "after every quota is filled."
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Optional manifest whose rows should not appear in SFT data.",
    )
    parser.add_argument("--exclude-split", default="countdown_eval")
    args = parser.parse_args()

    bucket_quotas = _parse_bucket_quotas(args.bucket_quotas)
    if bucket_quotas:
        records, metadata = build_records_with_bucket_quotas(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            local_jsonl=args.local_jsonl,
            bucket_quotas=bucket_quotas,
            max_scan_rows=args.max_scan_rows,
            traces_per_puzzle=args.traces_per_puzzle,
            prompt_style=args.prompt_style,
            trace_type=args.trace_type,
            min_numbers=args.min_numbers,
            max_numbers=args.max_numbers,
            solution_cap=args.solution_cap,
            exclude_manifest=args.exclude_manifest,
            exclude_split=args.exclude_split,
        )
    else:
        records, metadata = build_records(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            local_jsonl=args.local_jsonl,
            sample_size=args.sample_size,
            max_scan_rows=args.max_scan_rows,
            traces_per_puzzle=args.traces_per_puzzle,
            prompt_style=args.prompt_style,
            trace_type=args.trace_type,
            min_numbers=args.min_numbers,
            max_numbers=args.max_numbers,
            solution_cap=args.solution_cap,
            include_buckets=(
                tuple(args.include_buckets) if args.include_buckets else None
            ),
            exclude_manifest=args.exclude_manifest,
            exclude_split=args.exclude_split,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata["counts"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {len(records)} records to {args.output}")
    print(f"wrote metadata to {metadata_output}")


def build_records(
    *,
    dataset_name: str,
    dataset_split: str,
    local_jsonl: Path | None,
    sample_size: int,
    max_scan_rows: int,
    traces_per_puzzle: int,
    prompt_style: str,
    trace_type: str,
    min_numbers: int,
    max_numbers: int,
    solution_cap: int,
    include_buckets: tuple[str, ...] | None,
    exclude_manifest: Path | None,
    exclude_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Builds verified SFT prompt-completion rows from exact DFS solutions."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if max_scan_rows <= 0:
        raise ValueError("max_scan_rows must be positive")
    if traces_per_puzzle <= 0:
        raise ValueError("traces_per_puzzle must be positive")

    excluded_keys = _load_excluded_keys(exclude_manifest, exclude_split)
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    observed_solution_buckets: Counter[str] = Counter()
    scanned_rows = 0

    for dataset_index, row in _iter_rows(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        local_jsonl=local_jsonl,
    ):
        if scanned_rows >= max_scan_rows:
            break
        scanned_rows += 1

        try:
            numbers = _extract_numbers(row)
            target = _extract_target(row)
        except ValueError:
            skipped["bad_schema"] += 1
            continue
        if not min_numbers <= len(numbers) <= max_numbers:
            skipped["number_count_out_of_range"] += 1
            continue
        key = _puzzle_key(numbers, target)
        if key in excluded_keys:
            skipped["excluded_eval_key"] += 1
            continue

        count_result = count_countdown_solutions(numbers, target, cap=solution_cap)
        bucket = _bucket_for_count(count_result.count, solution_cap=solution_cap)
        observed_solution_buckets[bucket] += 1
        if count_result.count == 0:
            skipped["unsolvable"] += 1
            continue
        if include_buckets is not None and bucket not in include_buckets:
            skipped[f"bucket_{bucket}_excluded"] += 1
            continue

        solutions = solve_countdown_puzzle(
            numbers,
            target,
            max_solutions=traces_per_puzzle,
        )
        if not solutions:
            skipped["solver_no_solution_after_count"] += 1
            continue

        for solution_index, solution in enumerate(solutions):
            completion = format_completion(solution, trace_type=trace_type)
            verification = verify_answer(completion, solution.numbers, target=target)
            if not verification.valid:
                raise ValueError(
                    "generated invalid Countdown SFT completion for "
                    f"{numbers}, target={target}: {verification.reason}"
                )
            records.append(
                {
                    "id": (
                        f"countdown-sft-{dataset_index:07d}-"
                        f"{solution_index:02d}"
                    ),
                    "dataset_index": dataset_index,
                    "numbers": list(solution.numbers),
                    "target": target,
                    "prompt": format_prompt(
                        solution.numbers,
                        target=target,
                        prompt_style=prompt_style,
                    ),
                    "completion": completion,
                    "answer": solution.expression,
                    "trace": list(solution.trace),
                    "trace_type": trace_type,
                    "prompt_style": prompt_style,
                    "source": "countdown_exact_solver_sft",
                    "solution_count": count_result.count,
                    "solution_count_capped": count_result.capped,
                    "solution_count_bucket": _bucket_for_count(
                        count_result.count,
                        solution_cap=solution_cap,
                    ),
                }
            )
            if len(records) >= sample_size:
                break
        if len(records) >= sample_size:
            break

    if len(records) < sample_size:
        raise SystemExit(
            "insufficient verified Countdown SFT records: "
            f"requested={sample_size}, built={len(records)}, scanned={scanned_rows}"
        )

    metadata = {
        "schema_version": "countdown-solver-sft-data-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "local_jsonl": str(local_jsonl) if local_jsonl else None,
        "sample_size": sample_size,
        "max_scan_rows": max_scan_rows,
        "traces_per_puzzle": traces_per_puzzle,
        "prompt_style": prompt_style,
        "trace_type": trace_type,
        "solution_cap": solution_cap,
        "include_buckets": list(include_buckets) if include_buckets else None,
        "exclude_manifest": str(exclude_manifest) if exclude_manifest else None,
        "exclude_split": exclude_split,
        "verifier_version": VERIFIER_VERSION,
        "counts": {
            "records": len(records),
            "scanned_rows": scanned_rows,
            "excluded_keys": len(excluded_keys),
            "skipped": dict(sorted(skipped.items())),
            "observed_solution_count_buckets": dict(
                sorted(observed_solution_buckets.items())
            ),
        },
    }
    return records, metadata


def build_records_with_bucket_quotas(
    *,
    dataset_name: str,
    dataset_split: str,
    local_jsonl: Path | None,
    bucket_quotas: dict[str, int],
    max_scan_rows: int,
    traces_per_puzzle: int,
    prompt_style: str,
    trace_type: str,
    min_numbers: int,
    max_numbers: int,
    solution_cap: int,
    exclude_manifest: Path | None,
    exclude_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Builds verified SFT rows until every requested bucket quota is filled."""

    if not bucket_quotas:
        raise ValueError("bucket_quotas must not be empty")
    if any(quota <= 0 for quota in bucket_quotas.values()):
        raise ValueError(f"bucket quotas must be positive: {bucket_quotas}")
    if max_scan_rows <= 0:
        raise ValueError("max_scan_rows must be positive")
    if traces_per_puzzle <= 0:
        raise ValueError("traces_per_puzzle must be positive")

    excluded_keys = _load_excluded_keys(exclude_manifest, exclude_split)
    records: list[dict[str, Any]] = []
    records_by_bucket: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    observed_solution_buckets: Counter[str] = Counter()
    scanned_rows = 0

    for dataset_index, row in _iter_rows(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        local_jsonl=local_jsonl,
    ):
        if scanned_rows >= max_scan_rows:
            break
        if _quotas_filled(records_by_bucket, bucket_quotas):
            break
        scanned_rows += 1

        try:
            numbers = _extract_numbers(row)
            target = _extract_target(row)
        except ValueError:
            skipped["bad_schema"] += 1
            continue
        if not min_numbers <= len(numbers) <= max_numbers:
            skipped["number_count_out_of_range"] += 1
            continue
        key = _puzzle_key(numbers, target)
        if key in excluded_keys:
            skipped["excluded_eval_key"] += 1
            continue

        count_result = count_countdown_solutions(numbers, target, cap=solution_cap)
        bucket = _bucket_for_count(count_result.count, solution_cap=solution_cap)
        observed_solution_buckets[bucket] += 1
        if count_result.count == 0:
            skipped["unsolvable"] += 1
            continue
        if bucket not in bucket_quotas:
            skipped[f"bucket_{bucket}_excluded"] += 1
            continue
        remaining_quota = bucket_quotas[bucket] - records_by_bucket[bucket]
        if remaining_quota <= 0:
            skipped[f"bucket_{bucket}_filled"] += 1
            continue

        solutions = solve_countdown_puzzle(
            numbers,
            target,
            max_solutions=min(traces_per_puzzle, remaining_quota),
        )
        if not solutions:
            skipped["solver_no_solution_after_count"] += 1
            continue

        for solution_index, solution in enumerate(solutions):
            if records_by_bucket[bucket] >= bucket_quotas[bucket]:
                break
            completion = format_completion(solution, trace_type=trace_type)
            verification = verify_answer(completion, solution.numbers, target=target)
            if not verification.valid:
                raise ValueError(
                    "generated invalid Countdown SFT completion for "
                    f"{numbers}, target={target}: {verification.reason}"
                )
            records.append(
                {
                    "id": (
                        f"countdown-sft-{dataset_index:07d}-"
                        f"{solution_index:02d}"
                    ),
                    "dataset_index": dataset_index,
                    "numbers": list(solution.numbers),
                    "target": target,
                    "prompt": format_prompt(
                        solution.numbers,
                        target=target,
                        prompt_style=prompt_style,
                    ),
                    "completion": completion,
                    "answer": solution.expression,
                    "trace": list(solution.trace),
                    "trace_type": trace_type,
                    "prompt_style": prompt_style,
                    "source": "countdown_exact_solver_sft",
                    "solution_count": count_result.count,
                    "solution_count_capped": count_result.capped,
                    "solution_count_bucket": bucket,
                }
            )
            records_by_bucket[bucket] += 1

    missing = {
        bucket: quota - records_by_bucket[bucket]
        for bucket, quota in bucket_quotas.items()
        if records_by_bucket[bucket] < quota
    }
    if missing:
        raise SystemExit(
            "insufficient verified Countdown SFT records for bucket quotas: "
            f"missing={missing}, built={dict(records_by_bucket)}, "
            f"scanned={scanned_rows}"
        )

    metadata = {
        "schema_version": "countdown-solver-sft-data-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "local_jsonl": str(local_jsonl) if local_jsonl else None,
        "sample_size": len(records),
        "max_scan_rows": max_scan_rows,
        "traces_per_puzzle": traces_per_puzzle,
        "prompt_style": prompt_style,
        "trace_type": trace_type,
        "solution_cap": solution_cap,
        "bucket_quotas": dict(sorted(bucket_quotas.items())),
        "include_buckets": list(sorted(bucket_quotas)),
        "exclude_manifest": str(exclude_manifest) if exclude_manifest else None,
        "exclude_split": exclude_split,
        "verifier_version": VERIFIER_VERSION,
        "counts": {
            "records": len(records),
            "records_by_bucket": dict(sorted(records_by_bucket.items())),
            "scanned_rows": scanned_rows,
            "excluded_keys": len(excluded_keys),
            "skipped": dict(sorted(skipped.items())),
            "observed_solution_count_buckets": dict(
                sorted(observed_solution_buckets.items())
            ),
        },
    }
    return records, metadata


def _parse_bucket_quotas(raw_values: list[str] | None) -> dict[str, int]:
    if not raw_values:
        return {}
    quotas: dict[str, int] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise SystemExit(
                f"invalid --bucket-quotas item {raw_value!r}; expected bucket=quota"
            )
        bucket, quota_text = raw_value.split("=", 1)
        bucket = bucket.strip()
        if not bucket:
            raise SystemExit(f"invalid empty bucket in {raw_value!r}")
        try:
            quota = int(quota_text)
        except ValueError as exc:
            raise SystemExit(f"invalid quota in {raw_value!r}") from exc
        if quota <= 0:
            raise SystemExit(f"bucket quota must be positive in {raw_value!r}")
        quotas[bucket] = quota
    return quotas


def _quotas_filled(
    records_by_bucket: Counter[str],
    bucket_quotas: dict[str, int],
) -> bool:
    return all(
        records_by_bucket[bucket] >= quota
        for bucket, quota in bucket_quotas.items()
    )


def _iter_rows(
    *,
    dataset_name: str,
    dataset_split: str,
    local_jsonl: Path | None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if local_jsonl is not None:
        with local_jsonl.open(encoding="utf-8") as file:
            for index, line in enumerate(file):
                if line.strip():
                    yield index, json.loads(line)
        return

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on remote env.
        raise SystemExit(
            f"datasets is required without --local-jsonl: {exc}"
        ) from exc

    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
    for index, row in enumerate(dataset):
        yield index, row


def _load_excluded_keys(path: Path | None, split: str) -> set[str]:
    if path is None:
        return set()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        _puzzle_key(record["numbers"], record["target"])
        for record in manifest["splits"][split]
    }


def _puzzle_key(numbers: Iterable[int], target: int) -> str:
    return f"{int(target)}:{' '.join(str(int(number)) for number in sorted(numbers))}"


def _bucket_for_count(count: int, *, solution_cap: int) -> str:
    if count <= 0:
        return "0"
    if count >= solution_cap:
        return f"{solution_cap}_plus"
    return str(count)


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
