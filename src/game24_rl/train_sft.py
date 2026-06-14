"""LoRA SFT training entrypoint."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from game24_rl.data_gen import records_from_split_manifest, write_jsonl
from game24_rl.datasets import (
    DEFAULT_SPLIT_SEED,
    build_split_manifest,
    read_manifest,
    write_manifest,
)
from game24_rl.verifier import VERIFIER_VERSION

TRAINING_SCHEMA_VERSION = "game24-sft-run-v1"


@dataclass(frozen=True)
class SftDataConfig:
    """Data settings for first-pass SFT."""

    split_seed: int = DEFAULT_SPLIT_SEED
    traces_per_puzzle: int = 8
    trace_type: str = "short_success"
    prompt_style: str = "plain"
    target: int = 24
    train_split: str = "train"
    validation_split: str = "validation"
    manifest_path: str = "data/processed/splits/standard-game24-v1.json"
    train_jsonl_path: str = "data/processed/sft/game24-sft-v1-train.jsonl"


@dataclass(frozen=True)
class SftTrainingConfig:
    """Training settings for LoRA SFT."""

    output_dir: str = "outputs/sft_v1"
    run_name: str = "sft_v1"
    max_length: int = 512
    eos_token: str | None = "<|im_end|>"
    learning_rate: float = 1e-4
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    save_steps: int = 50
    logging_steps: int = 5
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    seed: int = 20260613
    bf16: bool = True
    report_to: list[str] = field(default_factory=lambda: ["none"])


@dataclass(frozen=True)
class SftEvaluationConfig:
    """Default evaluation settings attached to run metadata."""

    do_sample: bool = False
    max_new_tokens: int = 256
    success_gate: float = 0.70


@dataclass(frozen=True)
class SftRunConfig:
    """Complete SFT run configuration."""

    model_name: str
    method: str
    task: str
    data: SftDataConfig
    training: SftTrainingConfig
    evaluation: SftEvaluationConfig


def load_sft_config(path: str | Path) -> SftRunConfig:
    """Loads and validates a YAML SFT config."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    return SftRunConfig(
        model_name=_required_str(data, "model_name"),
        method=_required_str(data, "method"),
        task=_required_str(data, "task"),
        data=SftDataConfig(**data.get("data", {})),
        training=SftTrainingConfig(**data.get("training", {})),
        evaluation=SftEvaluationConfig(**data.get("evaluation", {})),
    )


def ensure_sft_inputs(config: SftRunConfig) -> tuple[Path, Path]:
    """Ensures the split manifest and SFT JSONL exist."""

    manifest_path = Path(config.data.manifest_path)
    train_jsonl_path = Path(config.data.train_jsonl_path)

    if not manifest_path.exists():
        manifest = build_split_manifest(
            seed=config.data.split_seed,
            target=config.data.target,
        )
        write_manifest(manifest, manifest_path)
    else:
        manifest = read_manifest(manifest_path)

    records = records_from_split_manifest(
        manifest,
        split=config.data.train_split,
        traces_per_puzzle=config.data.traces_per_puzzle,
        trace_type=config.data.trace_type,
        prompt_style=config.data.prompt_style,
    )
    if _needs_regeneration(train_jsonl_path, records):
        write_jsonl(records, train_jsonl_path)

    return manifest_path, train_jsonl_path


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Finds the latest ``checkpoint-*`` directory in a run directory."""

    path = Path(run_dir)
    if not path.exists():
        return None
    checkpoints = [item for item in path.glob("checkpoint-*") if item.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def resolve_resume_checkpoint(
    run_dir: str | Path,
    *,
    resume_from_checkpoint: str | None = None,
    auto_resume: bool = False,
) -> Path | None:
    """Resolves explicit or latest-checkpoint resume settings."""

    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        return checkpoint

    if auto_resume:
        return find_latest_checkpoint(run_dir)

    return None


def prepare_run_artifacts(
    *,
    config: SftRunConfig,
    config_path: str | Path,
    manifest_path: str | Path,
    train_jsonl_path: str | Path,
    dry_run: bool,
    resume_checkpoint: Path | None,
) -> dict[str, Path]:
    """Creates run directories and writes metadata."""

    run_dir = Path(config.training.output_dir) / config.training.run_name
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    for path in (run_dir, logs_dir, artifacts_dir):
        path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "dry_run": dry_run,
        "model_name": config.model_name,
        "method": config.method,
        "task": config.task,
        "verifier_version": VERIFIER_VERSION,
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "train_jsonl_path": str(train_jsonl_path),
        "run_dir": str(run_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "data": asdict(config.data),
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(config_path, artifacts_dir / "config.yaml")

    return {"run_dir": run_dir, "logs_dir": logs_dir, "artifacts_dir": artifacts_dir}


def run_sft(
    *,
    config_path: str | Path,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    auto_resume: bool = False,
) -> dict[str, Any]:
    """Runs or dry-runs the first-pass SFT pipeline."""

    config = load_sft_config(config_path)
    manifest_path, train_jsonl_path = ensure_sft_inputs(config)
    run_dir = Path(config.training.output_dir) / config.training.run_name
    resume_checkpoint = resolve_resume_checkpoint(
        run_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        auto_resume=auto_resume,
    )
    paths = prepare_run_artifacts(
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        train_jsonl_path=train_jsonl_path,
        dry_run=dry_run,
        resume_checkpoint=resume_checkpoint,
    )

    if dry_run:
        marker = paths["run_dir"] / "checkpoint-dry-run" / "metadata.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "dry_run": True,
                    "created_at": _utc_now(),
                    "message": "No model weights were loaded.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "dry_run_complete",
            "run_dir": str(paths["run_dir"]),
            "manifest_path": str(manifest_path),
            "train_jsonl_path": str(train_jsonl_path),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        }

    _run_real_sft(
        config=config,
        train_jsonl_path=train_jsonl_path,
        resume_checkpoint=resume_checkpoint,
    )
    return {
        "status": "training_complete",
        "run_dir": str(paths["run_dir"]),
        "manifest_path": str(manifest_path),
        "train_jsonl_path": str(train_jsonl_path),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
    }


def _run_real_sft(
    *,
    config: SftRunConfig,
    train_jsonl_path: Path,
    resume_checkpoint: Path | None,
) -> None:
    """Runs LoRA SFT with heavy training dependencies loaded lazily."""

    from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    from peft import LoraConfig  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from trl import SFTConfig, SFTTrainer  # pylint: disable=import-outside-toplevel

    run_dir = Path(config.training.output_dir) / config.training.run_name
    dataset = load_dataset("json", data_files=str(train_jsonl_path), split="train")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=config.training.lora_rank,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(run_dir),
        run_name=config.training.run_name,
        max_length=config.training.max_length,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        save_steps=config.training.save_steps,
        save_strategy="steps",
        logging_steps=config.training.logging_steps,
        logging_dir=str(run_dir / "logs"),
        seed=config.training.seed,
        bf16=config.training.bf16,
        report_to=config.training.report_to,
        eos_token=config.training.eos_token,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    trainer.save_model(str(run_dir / "final"))


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {key!r} must be a non-empty string")
    return value


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("checkpoint-")
    if suffix.isdigit():
        return (int(suffix), path.name)
    return (-1, path.name)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _needs_regeneration(train_jsonl_path: Path, records: list[dict[str, Any]]) -> bool:
    """Returns True when the cached SFT JSONL is missing or stale."""

    if not train_jsonl_path.exists():
        return True

    try:
        cached_records = [
            json.loads(line)
            for line in train_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError:
        return True

    if len(cached_records) != len(records):
        return True

    for cached, expected in zip(cached_records, records, strict=True):
        if cached != expected:
            return True

    return False
