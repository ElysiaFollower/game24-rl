"""Inspect one real SFT batch for label masking and loss sanity."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from game24_rl.data_gen import records_from_split_manifest, write_jsonl
from game24_rl.train_sft import load_sft_config


def main() -> None:
    """Builds a tiny SFT dataset and reports the supervised tokens/loss."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default="configs/sft_v1.yaml")
    parser.add_argument("--output", default="outputs/diagnostics/sft_batch.json")
    parser.add_argument("--train-limit", type=int, default=2)
    args = parser.parse_args()

    config = load_sft_config(args.config)
    records = _build_tiny_train_records(config, limit=args.train_limit)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        train_jsonl_path = tmpdir_path / "tiny-train.jsonl"
        config_path = tmpdir_path / "tiny-config.yaml"
        write_jsonl(records, train_jsonl_path)
        config_path.write_text(
            yaml.safe_dump(
                _build_probe_config(config, train_jsonl_path),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        summary = _inspect_batch(config_path=config_path, original_records=records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _build_tiny_train_records(config, *, limit: int) -> list[dict[str, Any]]:
    manifest = json.loads(Path(config.data.manifest_path).read_text(encoding="utf-8"))
    return records_from_split_manifest(
        manifest,
        split=config.data.train_split,
        traces_per_puzzle=1,
    )[:limit]


def _build_probe_config(config, train_jsonl_path: Path) -> dict[str, Any]:
    training = asdict(config.training)
    training.update(
        {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_train_epochs": 1,
            "save_steps": 999999,
            "logging_steps": 1,
            "bf16": False,
        }
    )
    return {
        "model_name": config.model_name,
        "method": config.method,
        "task": config.task,
        "data": {
            **asdict(config.data),
            "train_jsonl_path": str(train_jsonl_path),
        },
        "training": training,
        "evaluation": asdict(config.evaluation),
    }


def _inspect_batch(
    *,
    config_path: str | Path,
    original_records: list[dict[str, Any]],
) -> dict[str, Any]:
    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from peft import LoraConfig  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig, SFTTrainer  # pylint: disable=import-outside-toplevel

    config = load_sft_config(config_path)
    dataset = load_dataset(
        "json",
        data_files=config.data.train_jsonl_path,
        split="train",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(Path(config.training.output_dir) / "diagnose-sft-batch"),
            run_name="diagnose-sft-batch",
            max_length=config.training.max_length,
            learning_rate=config.training.learning_rate,
            num_train_epochs=config.training.num_train_epochs,
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            save_steps=config.training.save_steps,
            save_strategy="no",
            logging_steps=config.training.logging_steps,
            seed=config.training.seed,
            bf16=config.training.bf16,
            report_to=["none"],
            eos_token=config.training.eos_token,
            completion_only_loss=True,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )

    processed = trainer.train_dataset
    item = processed[0]
    collated = trainer.data_collator([item])
    collated = {
        key: value.to(trainer.model.device) if hasattr(value, "to") else value
        for key, value in collated.items()
    }
    with trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(trainer.model, collated)

    labels = collated["labels"][0].detach().cpu().tolist()
    input_ids = collated["input_ids"][0].detach().cpu().tolist()
    supervised_positions = [
        index for index, label in enumerate(labels) if label != -100
    ]
    supervised_ids = [labels[index] for index in supervised_positions]
    supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    first_record = original_records[0]
    completion = first_record["completion"]
    prompt = first_record["prompt"]

    return {
        "model_name": config.model_name,
        "records": len(original_records),
        "processed_columns": list(getattr(processed, "column_names", [])),
        "input_token_count": len(input_ids),
        "supervised_token_count": len(supervised_positions),
        "ignored_token_count": len(labels) - len(supervised_positions),
        "first_supervised_position": supervised_positions[0]
        if supervised_positions
        else None,
        "last_supervised_position": supervised_positions[-1]
        if supervised_positions
        else None,
        "loss": float(loss.detach().cpu()),
        "loss_is_finite": math.isfinite(float(loss.detach().cpu())),
        "prompt_token_count": len(tokenizer(prompt)["input_ids"]),
        "completion_token_count": len(tokenizer(completion)["input_ids"]),
        "full_text_prefix": full_text[:300],
        "supervised_text_prefix": supervised_text[:300],
        "expected_completion_prefix": completion[:300],
        "supervised_contains_answer": "<answer>" in supervised_text,
        "supervised_starts_with_completion": supervised_text.lstrip().startswith(
            completion[:20].lstrip()
        ),
        "prompt_leaked_into_supervised": prompt[:40] in supervised_text,
        "record_id": first_record["id"],
        "numbers": first_record["numbers"],
    }


if __name__ == "__main__":
    main()
