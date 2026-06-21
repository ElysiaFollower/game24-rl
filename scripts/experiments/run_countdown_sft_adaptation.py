"""Run a short SFT adaptation from the handoff2 Game24 GRPO adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from game24_rl.verifier import VERIFIER_VERSION  # noqa: E402

DEFAULT_BASE_MODEL = "outputs/experiments/baseline_format_v2_full_5000_from800/final"
DEFAULT_INITIAL_ADAPTER = (
    "outputs/experiments/handoff1_grpo_closure_control_smooth_v1_2000/"
    "train/checkpoint-500"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--initial-adapter", default=DEFAULT_INITIAL_ADAPTER)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--eos-token", default="<|im_end|>")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(args)
    (args.output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return

    run_training(args)
    print(f"wrote Countdown SFT adapter to {args.output_dir / 'final'}")


def build_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": "countdown-sft-adaptation-run-v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_jsonl": str(args.train_jsonl),
        "output_dir": str(args.output_dir),
        "base_model": args.base_model,
        "initial_adapter": args.initial_adapter,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "bf16": args.bf16,
        "eos_token": args.eos_token,
        "verifier_version": VERIFIER_VERSION,
        "objective": (
            "Teach the handoff2 Game24 GRPO adapter to respect arbitrary "
            "Countdown target prompts while preserving the solver-style trace."
        ),
    }


def run_training(args: argparse.Namespace) -> None:
    """Loads base SFT + handoff2 LoRA as trainable adapter and runs SFT."""

    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from peft import PeftModel  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig, SFTTrainer  # pylint: disable=import-outside-toplevel

    dataset = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.initial_adapter,
        is_trainable=True,
    )
    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        run_name=args.output_dir.name,
        max_length=args.max_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_steps=args.logging_steps,
        logging_dir=str(args.output_dir / "logs"),
        seed=args.seed,
        bf16=args.bf16,
        report_to=["none"],
        eos_token=args.eos_token,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
