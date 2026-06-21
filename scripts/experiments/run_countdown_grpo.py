"""Run GRPO adaptation on a sampled Countdown-Tasks-3to4 manifest."""

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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dataset-name", default="Jiayi-Pan/Countdown-Tasks-3to4")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--manifest-split", default="countdown_train")
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-style", default="qwen_chat_target")
    parser.add_argument("--rollout-num-generations", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--strict-pool-gates", action="store_true")
    parser.add_argument("--grpo-max-steps", type=int, default=100)
    parser.add_argument("--grpo-save-steps", type=int, default=50)
    parser.add_argument("--grpo-logging-steps", type=int, default=1)
    parser.add_argument("--grpo-max-completion-length", type=int, default=2048)
    parser.add_argument("--grpo-num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--scale-rewards", choices=["none", "group"], default="none")
    parser.add_argument(
        "--reward-profile",
        choices=["strict", "close_bonus", "closure_strict"],
        default="closure_strict",
    )
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = args.run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.run_root / "countdown-train-manifest.json"
    rollout_dir = args.run_root / "rollout"
    pool_manifest = args.run_root / "pool" / "pool-manifest.json"
    train_dir = args.run_root / "train"

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
        log_path=logs_dir / "00_build_manifest.log",
    )

    rollout_command = [
        sys.executable,
        "scripts/experiments/audit_rollout_distribution.py",
        "--manifest",
        str(manifest),
        "--split",
        args.manifest_split,
        "--checkpoint",
        args.start_checkpoint,
        "--model-name",
        args.model_name,
        "--training-mode",
        "full",
        "--prompt-style",
        args.prompt_style,
        "--output-dir",
        str(rollout_dir),
        "--num-generations",
        str(args.rollout_num_generations),
        "--batch-size",
        str(args.rollout_batch_size),
        "--max-new-tokens",
        str(args.rollout_max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        rollout_command.extend(["--limit", str(args.limit)])
    run(rollout_command, log_path=logs_dir / "01_rollout_audit.log")

    details = rollout_dir / f"{args.manifest_split}-rollout-details.json"
    pool_command = [
        sys.executable,
        "scripts/build_grpo_pool.py",
        "--details",
        str(details),
        "--output",
        str(pool_manifest),
        "--split",
        args.manifest_split,
        "--checkpoint",
        args.start_checkpoint,
        "--seed",
        str(args.seed),
        "--num-generations",
        str(args.rollout_num_generations),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.rollout_max_new_tokens),
        "--select-min-correct",
        "1",
        "--select-max-correct",
        str(max(1, args.rollout_num_generations - 1)),
    ]
    if not args.strict_pool_gates:
        pool_command.extend(
            [
                "--min-pool-size",
                "1",
                "--min-mixed-group-rate",
                "0.0",
                "--max-zero-std-group-rate",
                "1.0",
                "--min-correct-truncation-mixed",
                "0",
                "--max-all-wrong-rate",
                "1.0",
            ]
        )
    run(pool_command, log_path=logs_dir / "02_build_pool.log")

    common_grpo = [
        sys.executable,
        "scripts/train_grpo.py",
        "--manifest",
        str(manifest),
        "--split",
        args.manifest_split,
        "--output-dir",
        str(train_dir),
        "--prompt-style",
        args.prompt_style,
        "--model-name-or-path",
        args.start_checkpoint,
        "--pool-manifest",
        str(pool_manifest),
        "--max-steps",
        str(args.grpo_max_steps),
        "--save-steps",
        str(args.grpo_save_steps),
        "--logging-steps",
        str(args.grpo_logging_steps),
        "--max-completion-length",
        str(args.grpo_max_completion_length),
        "--num-generations",
        str(args.grpo_num_generations),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--learning-rate",
        str(args.learning_rate),
        "--beta",
        str(args.beta),
        "--scale-rewards",
        args.scale_rewards,
        "--reward-profile",
        args.reward_profile,
        "--peft-mode",
        "lora",
        "--no-mask-truncated-completions",
        "--no-remove-unused-columns",
    ]
    run([*common_grpo, "--compat-probe"], log_path=logs_dir / "03_compat_probe.log")
    run([*common_grpo, "--train"], log_path=logs_dir / "04_train_grpo.log")
    print(f"wrote Countdown GRPO model to {train_dir / 'final'}")


def write_metadata(args: argparse.Namespace, *, manifest: Path) -> None:
    metadata = {
        "schema_version": "countdown-grpo-wrapper-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "manifest": str(manifest),
        "manifest_split": args.manifest_split,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "start_checkpoint": args.start_checkpoint,
        "prompt_style": args.prompt_style,
        "rollout_num_generations": args.rollout_num_generations,
        "rollout_max_new_tokens": args.rollout_max_new_tokens,
        "grpo_max_steps": args.grpo_max_steps,
        "grpo_max_completion_length": args.grpo_max_completion_length,
        "reward_profile": args.reward_profile,
        "beta": args.beta,
        "scale_rewards": args.scale_rewards,
        "strict_pool_gates": args.strict_pool_gates,
    }
    (args.run_root / "run-metadata.json").write_text(
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
