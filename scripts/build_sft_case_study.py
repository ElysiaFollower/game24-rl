"""Build case-study artifacts for SFT validation failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from game24_rl.datasets import read_manifest
from game24_rl.evaluate import read_jsonl
from game24_rl.solver import solve_puzzle


def main() -> None:
    """Writes a compact case-study summary from eval and data artifacts."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/standard-game24-v1.json",
    )
    parser.add_argument(
        "--train-jsonl",
        default="data/processed/sft/game24-sft-v1-train.jsonl",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--raw-outputs", required=True)
    parser.add_argument("--output-dir", default="outputs/diagnostics/sft_case_study")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    train_records = read_jsonl(args.train_jsonl)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    raw_outputs = {record["id"]: record for record in read_jsonl(args.raw_outputs)}

    train_keys = {tuple(record["numbers"]) for record in train_records}
    train_by_key: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in train_records:
        train_by_key.setdefault(tuple(record["numbers"]), []).append(record)

    validation_keys = {
        tuple(record["numbers"]) for record in manifest["splits"][report["split"]]
    }
    overlap = sorted(train_keys & validation_keys)

    details = report["details"]
    failures = [detail for detail in details if not detail["valid"]]
    successes = [detail for detail in details if detail["valid"]]
    samples = [
        _sample_case(detail, raw_outputs, train_by_key)
        for detail in failures[: args.limit]
    ]

    summary = {
        "report": args.report,
        "raw_outputs": args.raw_outputs,
        "manifest": args.manifest,
        "train_jsonl": args.train_jsonl,
        "metrics": report["metrics"],
        "reason_counts": dict(Counter(detail["reason"] for detail in details)),
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_keys),
        "train_validation_overlap_count": len(overlap),
        "success_sample_count": len(successes),
        "failure_sample_count": len(failures),
        "failure_samples": samples,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _sample_case(
    detail: dict[str, Any],
    raw_outputs: dict[str, dict[str, Any]],
    train_by_key: dict[tuple[int, ...], list[dict[str, Any]]],
) -> dict[str, Any]:
    numbers = tuple(detail["numbers"])
    raw = raw_outputs.get(detail["id"], {})
    solver_solutions = solve_puzzle(numbers, max_solutions=3)
    similar_train = _nearest_train_records(numbers, train_by_key)
    return {
        "id": detail["id"],
        "numbers": detail["numbers"],
        "reason": detail["reason"],
        "model_expression": detail["expression"],
        "model_value": detail["value"],
        "raw_output": raw.get("output"),
        "solver_expressions": [solution.expression for solution in solver_solutions],
        "solver_traces": [list(solution.trace) for solution in solver_solutions],
        "nearest_train": similar_train,
    }


def _nearest_train_records(
    numbers: tuple[int, ...],
    train_by_key: dict[tuple[int, ...], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        train_by_key,
        key=lambda key: (_multiset_distance(numbers, key), key),
    )
    nearest = []
    for key in ranked[:3]:
        record = train_by_key[key][0]
        nearest.append(
            {
                "numbers": list(key),
                "distance": _multiset_distance(numbers, key),
                "answer": record["answer"],
                "completion": record["completion"],
            }
        )
    return nearest


def _multiset_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(abs(a - b) for a, b in zip(sorted(left), sorted(right), strict=True))


if __name__ == "__main__":
    main()
