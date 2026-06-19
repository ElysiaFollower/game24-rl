"""Summarize ToT full-eval reports into all/easy/hard groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-start", type=int, default=900)
    parser.add_argument("--hard-end", type=int, default=1000)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize_report(
        report,
        hard_start=args.hard_start,
        hard_end=args.hard_end,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["groups"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


def summarize_report(
    report: dict[str, Any],
    *,
    hard_start: int,
    hard_end: int,
) -> dict[str, Any]:
    details = list(report["details"])
    if len(details) != 1362:
        raise ValueError(
            "expected a single full ToT eval report with 1362 per-sample details, "
            f"got {len(details)}"
        )
    groups = {
        "all_1362": list(range(0, len(details))),
        "easy1262": [
            index
            for index in range(0, len(details))
            if not hard_start <= index < hard_end
        ],
        "hard100": list(range(hard_start, hard_end)),
    }
    return {
        "schema_version": "official-tot-group-summary-v1",
        "source_report": str(report.get("raw_outputs_path", "")),
        "eval_report_split": report.get("split"),
        "checkpoint": report.get("checkpoint"),
        "model_name": report.get("model_name"),
        "decoding": report.get("decoding"),
        "group_definitions": {
            "all_1362": "ToT indices 0-1361",
            "easy1262": "ToT indices 0-899 and 1000-1361",
            "hard100": "ToT indices 900-999",
        },
        "groups": {
            name: _summarize_details(details, indices)
            for name, indices in groups.items()
        },
    }


def _summarize_details(
    details: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    selected = [details[index] for index in indices]
    total = len(selected)
    solved = sum(int(item.get("valid", False)) for item in selected)
    format_ok = sum(int(item.get("format_ok", False)) for item in selected)
    valid_expr = sum(int(item.get("valid_expr", False)) for item in selected)
    reason_counts: dict[str, int] = {}
    for item in selected:
        reason = str(item.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "total": total,
        "solved": solved,
        "solve_rate": _safe_rate(solved, total),
        "format_ok": format_ok,
        "format_rate": _safe_rate(format_ok, total),
        "valid_expr": valid_expr,
        "valid_expr_rate": _safe_rate(valid_expr, total),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
