"""Build a stratified Countdown eval set and evaluate one model."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

MODEL_NAME = "outputs/experiments/baseline_format_v2_full_5000_from800/final"
CHECKPOINT = (
    "outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/"
    "train/checkpoint-500"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--training-mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--dataset-name", default="Jiayi-Pan/Countdown-Tasks-3to4")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--manifest-split", default="countdown_eval")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--buckets", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--max-scan-rows", type=int, default=200_000)
    parser.add_argument("--solution-cap", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prompt-style", default="qwen_chat_target")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "countdown-stratified-eval-manifest.json"
    write_metadata(args, manifest=manifest)

    build_command = [
        sys.executable,
        "scripts/build_countdown_stratified_eval_manifest.py",
        "--dataset-name",
        args.dataset_name,
        "--dataset-split",
        args.dataset_split,
        "--output",
        str(manifest),
        "--manifest-split",
        args.manifest_split,
        "--sample-size",
        str(args.sample_size),
        "--buckets",
        *args.buckets,
        "--max-scan-rows",
        str(args.max_scan_rows),
        "--solution-cap",
        str(args.solution_cap),
    ]
    if args.allow_partial:
        build_command.append("--allow-partial")
    run(build_command, log_path=args.output_dir / "build-manifest.log")

    run(
        [
            sys.executable,
            "scripts/eval_checkpoint.py",
            "--manifest",
            str(manifest),
            "--split",
            args.manifest_split,
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
    report = args.output_dir / f"{args.manifest_split}-eval-report.json"
    if not report.exists():
        raise SystemExit(f"expected eval report was not created: {report}")
    print(
        json.dumps(
            json.loads(report.read_text())["metrics"],
            indent=2,
            sort_keys=True,
        )
    )


def write_metadata(args: argparse.Namespace, *, manifest: Path) -> None:
    metadata = {
        "schema_version": "countdown-stratified-eval-wrapper-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "manifest": str(manifest),
        "manifest_split": args.manifest_split,
        "sample_size": args.sample_size,
        "buckets": args.buckets,
        "max_scan_rows": args.max_scan_rows,
        "solution_cap": args.solution_cap,
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "training_mode": args.training_mode,
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
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


if __name__ == "__main__":
    main()
