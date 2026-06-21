"""Build fast Countdown SFT data by finding one solution per unique puzzle."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.countdown_solver import solve_countdown_puzzle  # noqa: E402
from game24_rl.data_gen import format_completion, format_prompt  # noqa: E402
from game24_rl.verifier import VERIFIER_VERSION, verify_answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--max-scan-rows", type=int, default=490_364)
    parser.add_argument("--prompt-style", default="qwen_chat_minimal_target")
    parser.add_argument("--trace-type", default="short_success")
    parser.add_argument("--min-numbers", type=int, default=3)
    parser.add_argument("--max-numbers", type=int, default=4)
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--exclude-split", default="countdown_eval")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    records, metadata = build_records(args)
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


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_keys = _load_excluded_keys(args.exclude_manifest, args.exclude_split)
    seen_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    scanned = 0

    with args.local_jsonl.open(encoding="utf-8") as file:
        for dataset_index, line in enumerate(file):
            if scanned >= args.max_scan_rows or len(records) >= args.sample_size:
                break
            scanned += 1
            if not line.strip():
                skipped["blank_line"] += 1
                continue
            row = json.loads(line)
            try:
                numbers = [int(number) for number in row["nums"]]
                target = int(row["target"])
            except (KeyError, TypeError, ValueError):
                skipped["bad_schema"] += 1
                continue
            if not args.min_numbers <= len(numbers) <= args.max_numbers:
                skipped["number_count_out_of_range"] += 1
                continue
            key = f"{target}:{' '.join(str(number) for number in numbers)}"
            if key in excluded_keys:
                skipped["excluded_eval_key"] += 1
                continue
            if key in seen_keys:
                skipped["duplicate_key"] += 1
                continue

            solutions = solve_countdown_puzzle(numbers, target, max_solutions=1)
            if not solutions:
                skipped["unsolvable_or_not_found"] += 1
                continue
            solution = solutions[0]
            completion = format_completion(solution, trace_type=args.trace_type)
            verification = verify_answer(completion, solution.numbers, target=target)
            if not verification.valid:
                raise ValueError(
                    "generated invalid Countdown completion for "
                    f"{numbers}, target={target}: {verification.reason}"
                )
            seen_keys.add(key)
            records.append(
                {
                    "id": f"countdown-fast-sft-{dataset_index:07d}-00",
                    "dataset_index": dataset_index,
                    "numbers": list(solution.numbers),
                    "target": target,
                    "prompt": format_prompt(
                        solution.numbers,
                        target=target,
                        prompt_style=args.prompt_style,
                    ),
                    "completion": completion,
                    "answer": solution.expression,
                    "trace": list(solution.trace),
                    "trace_type": args.trace_type,
                    "prompt_style": args.prompt_style,
                    "source": "countdown_exact_solver_fast_sft",
                    "solution_count": None,
                    "solution_count_capped": None,
                    "solution_count_bucket": "unknown",
                }
            )
            if args.progress_every and scanned % args.progress_every == 0:
                print(
                    f"scanned={scanned} records={len(records)} "
                    f"skipped={dict(sorted(skipped.items()))}",
                    flush=True,
                )

    if len(records) < args.sample_size:
        raise SystemExit(
            "insufficient fast Countdown SFT records: "
            f"requested={args.sample_size}, built={len(records)}, scanned={scanned}"
        )

    metadata = {
        "schema_version": "countdown-fast-solver-sft-data-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "local_jsonl": str(args.local_jsonl),
        "sample_size": args.sample_size,
        "max_scan_rows": args.max_scan_rows,
        "prompt_style": args.prompt_style,
        "trace_type": args.trace_type,
        "exclude_manifest": str(args.exclude_manifest) if args.exclude_manifest else None,
        "exclude_split": args.exclude_split,
        "verifier_version": VERIFIER_VERSION,
        "counts": {
            "records": len(records),
            "scanned_rows": scanned,
            "excluded_keys": len(excluded_keys),
            "skipped": dict(sorted(skipped.items())),
        },
    }
    return records, metadata


def _load_excluded_keys(manifest_path: Path | None, split: str) -> set[str]:
    if manifest_path is None:
        return set()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = set()
    for row in manifest["splits"][split]:
        if "key" in row:
            keys.add(str(row["key"]))
        else:
            keys.add(
                f"{int(row['target'])}:"
                + " ".join(str(int(number)) for number in row["numbers"])
            )
    return keys


if __name__ == "__main__":
    main()
