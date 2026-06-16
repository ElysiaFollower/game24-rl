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
from game24_rl.grpo import (
    GrpoCompatibilityConfig,
    GrpoPoolGateConfig,
    audit_rollout_details_file,
    build_grpo_probe_metadata,
    build_prompt_dataset_from_pool,
    run_grpo_dry_run,
)
from game24_rl.rewards import reward_completions
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
    parser.add_argument("--trace-type", default="short_success")
    parser.add_argument("--prompt-style", default="plain")
    args = parser.parse_args()

    manifest = read_split_manifest(args.manifest)
    records = records_from_split_manifest(
        manifest,
        split=args.split,
        traces_per_puzzle=args.traces_per_puzzle,
        trace_type=args.trace_type,
        prompt_style=args.prompt_style,
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
    parser.add_argument("--prompt-style", default="plain")
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
            prompt_style=args.prompt_style,
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
                prompt_style=args.prompt_style,
            )
        report = evaluate_raw_outputs_file(
            raw_outputs_path=raw_outputs,
            report_path=output_dir / f"{args.split}-eval-report.json",
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            split_manifest=args.manifest,
            split=args.split,
            decoding=decoding,
            generation_prompt_style=args.prompt_style,
        )

    metrics = report["metrics"]
    print(
        "evaluated {total} records: solve_rate={solve_rate:.3f}, "
        "format_rate={format_rate:.3f}, valid_expr_rate={valid_expr_rate:.3f}".format(
            **metrics
        )
    )


def train_grpo_main() -> None:
    """Runs safe GRPO preparation steps or fails fast before real training."""

    parser = argparse.ArgumentParser(description=train_grpo_main.__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/splits/standard-game24-v1.json",
        help="Split manifest path.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/grpo_pilot_v1",
        help="Directory for GRPO artifacts.",
    )
    parser.add_argument("--prompt-style", default="qwen_chat")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compat-probe", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--pool-manifest")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--scale-rewards", choices=["none", "group"], default="none")
    parser.add_argument(
        "--mask-truncated-completions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--remove-unused-columns",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if args.dry_run:
        result = run_grpo_dry_run(
            manifest_path=args.manifest,
            split=args.split,
            output_dir=args.output_dir,
            prompt_style=args.prompt_style,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.compat_probe:
        result = _run_grpo_compat_probe(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return

    if args.train:
        result = _run_real_grpo(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raise SystemExit(
        "Specify one of --dry-run, --compat-probe, or --train. Run --dry-run "
        "and --compat-probe before --train."
    )


def build_grpo_pool_main() -> None:
    """Audits sampled rollout details and writes a GRPO pool manifest."""

    parser = argparse.ArgumentParser(description=build_grpo_pool_main.__doc__)
    parser.add_argument("--details", required=True, help="Rollout details JSON path.")
    parser.add_argument("--output", required=True, help="Output pool manifest JSON.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint", help="Checkpoint used for rollout sampling.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-generations", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--min-pool-size", type=int, default=200)
    parser.add_argument("--min-mixed-group-rate", type=float, default=0.25)
    parser.add_argument("--max-zero-std-group-rate", type=float, default=0.75)
    parser.add_argument("--min-correct-truncation-mixed", type=int, default=50)
    parser.add_argument("--max-all-wrong-rate", type=float, default=0.25)
    args = parser.parse_args()

    gate = GrpoPoolGateConfig(
        min_pool_size=args.min_pool_size,
        min_mixed_group_rate=args.min_mixed_group_rate,
        max_zero_std_group_rate=args.max_zero_std_group_rate,
        min_correct_truncation_mixed=args.min_correct_truncation_mixed,
        max_all_wrong_rate=args.max_all_wrong_rate,
    )
    metadata = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    audit = audit_rollout_details_file(
        args.details,
        args.output,
        gate=gate,
        metadata=metadata,
    )
    print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
    if not audit.passed:
        raise SystemExit(1)


def _run_grpo_compat_probe(args: argparse.Namespace) -> dict[str, object]:
    try:
        import trl  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
        from trl import GRPOConfig  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        return {
            "schema_version": "game24-grpo-compat-probe-v1",
            "passed": False,
            "error": f"TRL is not installed or cannot be imported: {exc}",
        }

    supported_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
    metadata = build_grpo_probe_metadata(
        GrpoCompatibilityConfig(
            beta=args.beta,
            scale_rewards=args.scale_rewards,
            mask_truncated_completions=args.mask_truncated_completions,
            remove_unused_columns=args.remove_unused_columns,
        ),
        trl_version=getattr(trl, "__version__", None),
        supported_fields=supported_fields,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata["passed"]:
        try:
            resolved_config = GRPOConfig(
                output_dir=str(output_dir),
                beta=args.beta,
                scale_rewards=args.scale_rewards,
                mask_truncated_completions=args.mask_truncated_completions,
                remove_unused_columns=args.remove_unused_columns,
                loss_type="dr_grpo",
            )
        except Exception as exc:  # pragma: no cover - depends on remote TRL version.
            metadata["passed"] = False
            metadata["error"] = f"GRPOConfig rejected probe settings: {exc}"
        else:
            metadata["resolved_config"] = {
                "beta": getattr(resolved_config, "beta", None),
                "scale_rewards": getattr(resolved_config, "scale_rewards", None),
                "mask_truncated_completions": getattr(
                    resolved_config,
                    "mask_truncated_completions",
                    None,
                ),
                "remove_unused_columns": getattr(
                    resolved_config,
                    "remove_unused_columns",
                    None,
                ),
                "loss_type": getattr(resolved_config, "loss_type", None),
            }
    (output_dir / "compat-probe.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _run_real_grpo(args: argparse.Namespace) -> dict[str, object]:
    if not args.model_name_or_path:
        raise SystemExit("--model-name-or-path is required for --train")
    if not args.pool_manifest:
        raise SystemExit("--pool-manifest is required for --train")

    try:
        from datasets import (  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
            Dataset,
        )
        from trl import (  # type: ignore[import-not-found]  # noqa: PLC0415
            GRPOConfig,
            GRPOTrainer,
        )
    except ImportError as exc:
        raise SystemExit(f"GRPO training dependencies are unavailable: {exc}") from exc

    prompt_records = build_prompt_dataset_from_pool(
        manifest_path=args.manifest,
        pool_manifest_path=args.pool_manifest,
        split=args.split,
        prompt_style=args.prompt_style,
        limit=args.limit,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_kwargs, skipped_config_kwargs = _build_grpo_config_kwargs(
        supported_fields=set(getattr(GRPOConfig, "__dataclass_fields__", {})),
        output_dir=str(output_dir),
        beta=args.beta,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=args.mask_truncated_completions,
        remove_unused_columns=args.remove_unused_columns,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    config = GRPOConfig(**config_kwargs)
    metadata = {
        "schema_version": "game24-grpo-train-run-v1",
        "model_name_or_path": args.model_name_or_path,
        "manifest": args.manifest,
        "split": args.split,
        "pool_manifest": args.pool_manifest,
        "prompt_records": len(prompt_records),
        "output_dir": str(output_dir),
        "beta": args.beta,
        "scale_rewards": args.scale_rewards,
        "mask_truncated_completions": args.mask_truncated_completions,
        "remove_unused_columns": args.remove_unused_columns,
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_prompt_length_requested": args.max_prompt_length,
        "max_prompt_length_applied_by_trl": False,
        "grpo_config_kwargs": config_kwargs,
        "skipped_grpo_config_kwargs": skipped_config_kwargs,
    }
    (output_dir / "train-run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=reward_completions,
        args=config,
        train_dataset=Dataset.from_list(prompt_records),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    metadata["status"] = "complete"
    (output_dir / "train-run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _build_grpo_config_kwargs(
    *,
    supported_fields: set[str],
    output_dir: str,
    beta: float,
    scale_rewards: str,
    mask_truncated_completions: bool,
    remove_unused_columns: bool,
    max_completion_length: int,
    num_generations: int,
    temperature: float,
    top_p: float,
    learning_rate: float,
    max_steps: int,
    save_steps: int,
    logging_steps: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Builds GRPOConfig kwargs for the installed TRL version.

    TRL 1.6.0 no longer exposes ``max_prompt_length``. Prompts in this project
    are short, so prompt limiting is recorded in metadata instead of being
    forced through an unsupported config field.
    """

    required = {
        "output_dir": output_dir,
        "beta": beta,
        "scale_rewards": scale_rewards,
        "mask_truncated_completions": mask_truncated_completions,
        "remove_unused_columns": remove_unused_columns,
        "loss_type": "dr_grpo",
        "max_completion_length": max_completion_length,
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "bf16": True,
        "report_to": ["tensorboard"],
    }
    optional = {
        "temperature": temperature,
        "top_p": top_p,
    }

    missing_required = sorted(set(required) - supported_fields)
    if missing_required:
        raise SystemExit(
            "Installed TRL GRPOConfig does not support required fields: "
            + ", ".join(missing_required)
        )

    config_kwargs = dict(required)
    skipped = {}
    for name, value in optional.items():
        if name in supported_fields:
            config_kwargs[name] = value
        else:
            skipped[name] = value
    return config_kwargs, skipped
