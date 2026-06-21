"""Sample Countdown-Tasks-3to4 and evaluate one model with target-aware prompts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--dataset-name", default="Jiayi-Pan/Countdown-Tasks-3to4")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--manifest-split", default="countdown_sample")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prompt-style", default="qwen_chat_target")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "countdown-sample-manifest.json"
    write_metadata(args, manifest=manifest)

    run(
        [
            sys.executable,
            "scripts/build_countdown_manifest.py",
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
            "--seed",
            str(args.seed),
        ],
        log_path=args.output_dir / "build-manifest.log",
    )
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
    print(json.dumps(json.loads(report.read_text())["metrics"], indent=2, sort_keys=True))


def write_metadata(args: argparse.Namespace, *, manifest: Path) -> None:
    metadata = {
        "schema_version": "countdown-sample-eval-wrapper-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "manifest": str(manifest),
        "manifest_split": args.manifest_split,
        "sample_size": args.sample_size,
        "seed": args.seed,
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
