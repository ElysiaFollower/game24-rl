"""Evaluate one model on Game24 and optionally write ToT group summaries."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MANIFEST = "data/processed/splits/official-tot-overnight-v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--split", default="tot_all_1362")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prompt-style", default="qwen_chat")
    parser.add_argument("--no-group-summary", action="store_true")
    parser.add_argument("--hard-start", type=int, default=900)
    parser.add_argument("--hard-end", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args)
    report = args.output_dir / f"{args.split}-eval-report.json"

    run(
        [
            sys.executable,
            "scripts/eval_checkpoint.py",
            "--manifest",
            args.manifest,
            "--split",
            args.split,
            "--output-dir",
            str(args.output_dir),
            "--model-name",
            args.model_name,
            "--checkpoint",
            args.checkpoint,
            "--training-mode",
            args.training_mode,
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--prompt-style",
            args.prompt_style,
        ],
        log_path=args.output_dir / "eval.log",
    )
    if not report.exists():
        raise SystemExit(f"expected eval report was not created: {report}")

    if not args.no_group_summary:
        run(
            [
                sys.executable,
                "scripts/summarize_tot_eval_groups.py",
                "--report",
                str(report),
                "--output",
                str(args.output_dir / "group-summary.json"),
                "--hard-start",
                str(args.hard_start),
                "--hard-end",
                str(args.hard_end),
            ],
            log_path=args.output_dir / "group-summary.log",
        )

    print_report_summary(report)


def write_metadata(args: argparse.Namespace) -> None:
    metadata = {
        "schema_version": "game24-model-eval-wrapper-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": args.manifest,
        "split": args.split,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "training_mode": args.training_mode,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
        "writes_group_summary": not args.no_group_summary,
    }
    (args.output_dir / "eval-wrapper-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(command: list[str], *, log_path: Path) -> None:
    command_text = " ".join(shlex.quote(part) for part in command)
    print(f"[run] {command_text}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {command_text}\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )


def print_report_summary(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
