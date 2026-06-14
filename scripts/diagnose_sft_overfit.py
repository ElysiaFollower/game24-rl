"""Run a tiny SFT overfit smoke to validate the training pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml

from game24_rl.data_gen import records_from_split_manifest, write_jsonl
from game24_rl.evaluate import DecodingConfig, evaluate_raw_outputs_file
from game24_rl.train_sft import find_latest_checkpoint, load_sft_config, run_sft


def main() -> None:
    """Trains on a tiny subset and checks memorization on the same records."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default="configs/sft_v1.yaml")
    parser.add_argument("--output-dir", default="outputs/diagnostics/sft_overfit")
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=12.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory before running the smoke.",
    )
    args = parser.parse_args()

    config = load_sft_config(args.config)
    output_dir = Path(args.output_dir)
    if args.force and output_dir.exists():
        import shutil  # pylint: disable=import-outside-toplevel

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        train_jsonl_path = tmpdir_path / "tiny-train.jsonl"
        tiny_records = _build_tiny_train_records(config, limit=args.train_limit)
        write_jsonl(tiny_records, train_jsonl_path)

        tiny_config = _build_tiny_config(
            config_path=Path(args.config),
            config=config,
            output_dir=output_dir,
            train_jsonl_path=train_jsonl_path,
            epochs=args.epochs,
        )
        tiny_config_path = tmpdir_path / "tiny-config.yaml"
        tiny_config_path.write_text(
            yaml.safe_dump(tiny_config, sort_keys=False),
            encoding="utf-8",
        )

        result = run_sft(config_path=tiny_config_path, dry_run=False)
        run_dir = Path(result["run_dir"])
        latest_checkpoint = find_latest_checkpoint(run_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"no checkpoint produced in {run_dir}")

        raw_outputs_path = output_dir / "train-raw-outputs.jsonl"
        report_path = output_dir / "train-eval-report.json"
        _generate_outputs_for_records(
            records=tiny_records,
            output_path=raw_outputs_path,
            model_name=config.model_name,
            checkpoint=latest_checkpoint,
            decoding=DecodingConfig(max_new_tokens=args.max_new_tokens),
        )
        report = evaluate_raw_outputs_file(
            raw_outputs_path=raw_outputs_path,
            report_path=report_path,
            model_name=config.model_name,
            checkpoint=str(latest_checkpoint),
            split_manifest=config.data.manifest_path,
            split=config.data.train_split,
            decoding=DecodingConfig(max_new_tokens=args.max_new_tokens),
        )

        state_path = latest_checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        losses = [
            record["loss"]
            for record in state.get("log_history", [])
            if "loss" in record
        ]

        summary = {
            "status": result["status"],
            "run_dir": str(run_dir),
            "checkpoint": str(latest_checkpoint),
            "train_limit": args.train_limit,
            "epochs": args.epochs,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_records": len(losses),
            "metrics": report["metrics"],
            "raw_outputs_path": str(raw_outputs_path),
            "report_path": str(report_path),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


def _build_tiny_train_records(config, *, limit: int) -> list[dict]:
    manifest = _load_manifest(config.data.manifest_path)
    return records_from_split_manifest(
        manifest,
        split=config.data.train_split,
        traces_per_puzzle=1,
    )[:limit]


def _load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _generate_outputs_for_records(
    *,
    records: list[dict],
    output_path: str | Path,
    model_name: str,
    checkpoint: str | Path,
    decoding: DecodingConfig,
) -> None:
    from peft import PeftModel  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model.eval()

    raw_outputs = []
    for record in records:
        prompt = record["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            do_sample=decoding.do_sample,
            max_new_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        raw_outputs.append(
            {
                "id": record["id"],
                "numbers": record["numbers"],
                "target": record["target"],
                "prompt": prompt,
                "output": output,
                "source": "tiny_overfit_checkpoint_generation",
            }
        )

    write_jsonl(raw_outputs, output_path)


def _build_tiny_config(
    *,
    config_path: Path,
    config,
    output_dir: Path,
    train_jsonl_path: Path,
    epochs: float,
) -> dict:
    return {
        "model_name": config.model_name,
        "method": config.method,
        "task": config.task,
        "data": {
            **asdict_like(config.data),
            "train_jsonl_path": str(train_jsonl_path),
        },
        "training": {
            **asdict_like(config.training),
            "output_dir": str(output_dir),
            "run_name": "tiny-overfit",
            "save_steps": 1,
            "logging_steps": 1,
            "gradient_accumulation_steps": 1,
            "per_device_train_batch_size": 1,
            "num_train_epochs": epochs,
        },
        "evaluation": asdict_like(config.evaluation),
        "source_config": str(config_path),
    }


def asdict_like(obj) -> dict:
    """Converts a frozen dataclass-like object to a plain dict."""

    return asdict(obj)


if __name__ == "__main__":
    main()
