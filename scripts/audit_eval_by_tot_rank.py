"""Audit raw outputs or eval reports by Tree-of-Thoughts rank buckets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game24_rl.verifier import VERIFIER_VERSION, verify_answer  # noqa: E402


DEFAULT_TOT_MANIFEST = Path("data/processed/splits/official-tot-overnight-v1.json")
DEFAULT_TOT_SPLIT = "tot_all_1362"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tot-manifest", type=Path, default=DEFAULT_TOT_MANIFEST)
    parser.add_argument("--tot-split", default=DEFAULT_TOT_SPLIT)
    parser.add_argument("--report", type=Path, action="append")
    parser.add_argument("--raw-outputs", type=Path, action="append")
    parser.add_argument("--name", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = collect_inputs(args)
    if args.name and len(args.name) != len(inputs):
        raise SystemExit("--name must be repeated once per input")

    rank_index = build_tot_rank_index(args.tot_manifest, args.tot_split)
    reports = {}
    for index, (source_type, source_path) in enumerate(inputs):
        name = args.name[index] if args.name else source_path.stem
        if source_type == "report":
            reports[name] = audit_report(source_path, rank_index)
        else:
            reports[name] = audit_raw_outputs(source_path, rank_index)

    payload = {
        "schema_version": "eval-by-tot-rank-audit-v1",
        "tot_manifest": str(args.tot_manifest),
        "tot_split": args.tot_split,
        "rank_source": "matched by sorted input-number multiset to ToT manifest tot_index + 1",
        "verifier_version_for_raw_outputs": VERIFIER_VERSION,
        "bucket_definitions": {
            "easy_1_300": "ToT rank 1-300",
            "medium_301_900": "ToT rank 301-900",
            "hard_901_1100": "ToT rank 901-1100",
            "very_hard_1101_1362": "ToT rank 1101-1362",
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


def collect_inputs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    inputs.extend(("report", path) for path in args.report or [])
    inputs.extend(("raw_outputs", path) for path in args.raw_outputs or [])
    if not inputs:
        raise SystemExit("provide at least one --report or --raw-outputs")
    return inputs


def build_tot_rank_index(manifest_path: Path, split: str) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["splits"][split]
    index: dict[str, dict[str, Any]] = {}
    for fallback_index, record in enumerate(records):
        key = puzzle_key(record["numbers"])
        rank = int(record.get("tot_index", fallback_index)) + 1
        if key in index:
            raise ValueError(f"duplicate ToT puzzle key in manifest: {key}")
        index[key] = {
            "rank": rank,
            "tot_index": rank - 1,
            "tot_id": record.get("id"),
            "tot_solved_rate": record.get("tot_solved_rate"),
        }
    return index


def audit_report(
    report_path: Path,
    rank_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    buckets: dict[str, dict[str, Any]] = {
        name: {"total": 0, "solved": 0, "format_ok": 0, "valid_expr": 0}
        for name in bucket_order()
    }
    reason_counts: dict[str, Counter[str]] = {name: Counter() for name in bucket_order()}
    missing_keys = []

    for detail in report["details"]:
        key = puzzle_key(detail["numbers"])
        rank_info = rank_index.get(key)
        if rank_info is None:
            missing_keys.append(key)
            continue
        bucket = bucket_for_rank(int(rank_info["rank"]))
        item = buckets[bucket]
        item["total"] += 1
        item["solved"] += int(detail.get("valid", False))
        item["format_ok"] += int(detail.get("format_ok", False))
        item["valid_expr"] += int(detail.get("valid_expr", False))
        reason_counts[bucket][str(detail.get("reason", "unknown"))] += 1

    if missing_keys:
        raise ValueError(
            f"{report_path} has puzzles not present in ToT manifest: {missing_keys[:10]}"
        )

    return {
        "source_type": "eval_report",
        "report": str(report_path),
        "checkpoint": report.get("checkpoint"),
        "split_manifest": report.get("split_manifest"),
        "split": report.get("split"),
        "decoding": report.get("decoding"),
        "generation_prompt_style": report.get("generation_prompt_style"),
        "overall_metrics": report.get("metrics"),
        "buckets": {
            name: summarize_bucket(buckets[name], reason_counts[name])
            for name in bucket_order()
        },
    }


def audit_raw_outputs(
    raw_outputs_path: Path,
    rank_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_records = read_jsonl(raw_outputs_path)
    buckets: dict[str, dict[str, Any]] = {
        name: {"total": 0, "solved": 0, "format_ok": 0, "valid_expr": 0}
        for name in bucket_order()
    }
    reason_counts: dict[str, Counter[str]] = {name: Counter() for name in bucket_order()}
    missing_keys = []

    for record in raw_records:
        key = puzzle_key(record["numbers"])
        rank_info = rank_index.get(key)
        if rank_info is None:
            missing_keys.append(key)
            continue
        result = verify_answer(
            record["output"],
            puzzle=record["numbers"],
            target=record.get("target", 24),
        )
        format_ok = not result.reason.startswith("answer_contract")
        valid_expr = result.valid or result.reason == "wrong_value"
        bucket = bucket_for_rank(int(rank_info["rank"]))
        item = buckets[bucket]
        item["total"] += 1
        item["solved"] += int(result.valid)
        item["format_ok"] += int(format_ok)
        item["valid_expr"] += int(valid_expr)
        reason_counts[bucket][result.reason] += 1

    if missing_keys:
        raise ValueError(
            f"{raw_outputs_path} has puzzles not present in ToT manifest: "
            f"{missing_keys[:10]}"
        )

    overall = summarize_records_from_buckets(buckets)
    return {
        "source_type": "raw_outputs_reverified",
        "raw_outputs": str(raw_outputs_path),
        "verifier_version": VERIFIER_VERSION,
        "overall_metrics": overall,
        "buckets": {
            name: summarize_bucket(buckets[name], reason_counts[name])
            for name in bucket_order()
        },
    }


def summarize_records_from_buckets(
    buckets: dict[str, dict[str, int]],
) -> dict[str, int | float]:
    total = sum(bucket["total"] for bucket in buckets.values())
    solved = sum(bucket["solved"] for bucket in buckets.values())
    format_ok = sum(bucket["format_ok"] for bucket in buckets.values())
    valid_expr = sum(bucket["valid_expr"] for bucket in buckets.values())
    return {
        "total": total,
        "format_ok": format_ok,
        "format_rate": safe_rate(format_ok, total),
        "valid_expr": valid_expr,
        "valid_expr_rate": safe_rate(valid_expr, total),
        "solved": solved,
        "solve_rate": safe_rate(solved, total),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_bucket(
    bucket: dict[str, int],
    reasons: Counter[str],
) -> dict[str, Any]:
    total = bucket["total"]
    return {
        "total": total,
        "solved": bucket["solved"],
        "solve_rate": safe_rate(bucket["solved"], total),
        "format_ok": bucket["format_ok"],
        "format_rate": safe_rate(bucket["format_ok"], total),
        "valid_expr": bucket["valid_expr"],
        "valid_expr_rate": safe_rate(bucket["valid_expr"], total),
        "reason_counts": dict(sorted(reasons.items())),
    }


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


def bucket_order() -> list[str]:
    return [
        "easy_1_300",
        "medium_301_900",
        "hard_901_1100",
        "very_hard_1101_1362",
    ]


def puzzle_key(numbers: list[int]) -> str:
    return " ".join(str(number) for number in sorted(int(number) for number in numbers))


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
