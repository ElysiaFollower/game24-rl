"""Console-script entrypoints for Game24 utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game24_rl.data_gen import records_from_split_manifest, write_jsonl
from game24_rl.datasets import DEFAULT_SPLIT_SEED, build_split_manifest, write_manifest
from game24_rl.datasets import read_manifest as read_split_manifest
from game24_rl.evaluate import (
    DecodingConfig,
    evaluate_raw_outputs_file,
    evaluate_solver_dry_run,
    generate_checkpoint_outputs,
)
from game24_rl.train_sft import run_sft


def make_splits_main() -> None:
    """Creates a deterministic standard Game24 split manifest."""

    parser = argparse.ArgumentParser(
        description="Create a deterministic standard Game24 split manifest.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/splits/standard-game24-v1.json",
        help="Output manifest path.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    manifest = build_split_manifest(
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    write_manifest(manifest, args.output)
    counts = manifest["counts"]
    print(
        "wrote {output}: total={total}, solvable={solvable}, "
        "unsolvable={unsolvable}, train={train}, validation={validation}, "
        "test={test}".format(output=args.output, **counts)
    )


def build_sft_v1_main() -> None:
    """Builds first-pass Game24 short-success-trace SFT JSONL."""

    parser = argparse.ArgumentParser(
        description="Build first-pass Game24 short-success-trace SFT JSONL.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/standard-game24-v1.json",
        help="Split manifest produced by game24-make-splits.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/sft/game24-sft-v1-train.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--traces-per-puzzle", type=int, default=8)
    args = parser.parse_args()

    manifest = read_split_manifest(args.manifest)
    records = records_from_split_manifest(
        manifest,
        split=args.split,
        traces_per_puzzle=args.traces_per_puzzle,
    )
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")


def train_sft_main() -> None:
    """Runs or dry-runs the first-pass LoRA SFT pipeline."""

    parser = argparse.ArgumentParser(description=train_sft_main.__doc__)
    parser.add_argument("--config", default="configs/sft_v1.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data and artifacts without loading model weights.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Explicit checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from latest checkpoint in output_dir/run_name if present.",
    )
    args = parser.parse_args()

    result = run_sft(
        config_path=args.config,
        dry_run=args.dry_run,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def eval_checkpoint_main() -> None:
    """Evaluates Game24 raw outputs or a LoRA checkpoint."""

    parser = argparse.ArgumentParser(description=eval_checkpoint_main.__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/standard-game24-v1.json",
        help="Split manifest path.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--output-dir",
        default="outputs/eval",
        help="Directory for raw outputs and report artifacts.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", help="LoRA checkpoint path.")
    parser.add_argument("--raw-outputs", help="Existing raw-output JSONL to score.")
    parser.add_argument(
        "--solver-dry-run",
        action="store_true",
        help="Use exact solver outputs without loading model weights.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoding = DecodingConfig(
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.solver_dry_run:
        report = evaluate_solver_dry_run(
            manifest_path=args.manifest,
            output_dir=output_dir,
            split=args.split,
            limit=args.limit or 16,
            model_name="exact-solver-dry-run",
        )
    else:
        raw_outputs = Path(args.raw_outputs) if args.raw_outputs else None
        if raw_outputs is None:
            if not args.checkpoint:
                raise SystemExit(
                    "Either --solver-dry-run, --raw-outputs, "
                    "or --checkpoint is required."
                )
            raw_outputs = output_dir / f"{args.split}-raw-outputs.jsonl"
            generate_checkpoint_outputs(
                manifest_path=args.manifest,
                split=args.split,
                output_path=raw_outputs,
                model_name=args.model_name,
                checkpoint=args.checkpoint,
                decoding=decoding,
                limit=args.limit,
            )
        report = evaluate_raw_outputs_file(
            raw_outputs_path=raw_outputs,
            report_path=output_dir / f"{args.split}-eval-report.json",
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            split_manifest=args.manifest,
            split=args.split,
            decoding=decoding,
        )

    metrics = report["metrics"]
    print(
        "evaluated {total} records: solve_rate={solve_rate:.3f}, "
        "format_rate={format_rate:.3f}, valid_expr_rate={valid_expr_rate:.3f}".format(
            **metrics
        )
    )
