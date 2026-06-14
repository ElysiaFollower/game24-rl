"""Tests for SFT run preparation and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from game24_rl.datasets import build_split_manifest, write_manifest
from game24_rl.evaluate import (
    DecodingConfig,
    evaluate_raw_outputs_file,
    evaluate_solver_dry_run,
    write_jsonl,
)
from game24_rl.train_sft import (
    find_latest_checkpoint,
    load_sft_config,
    resolve_resume_checkpoint,
    run_sft,
)
from game24_rl.verifier import VERIFIER_VERSION


def test_load_sft_config_includes_resume_safe_defaults() -> None:
    config = load_sft_config("configs/sft_v1.yaml")

    assert config.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.data.split_seed == 20260613
    assert config.training.max_length == 512
    assert config.training.eos_token == "<|im_end|>"
    assert config.training.save_steps == 50
    assert config.training.output_dir == "outputs/sft_v1"


def test_latest_checkpoint_uses_numeric_suffix(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-final").mkdir()

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-10"
    assert (
        resolve_resume_checkpoint(tmp_path, auto_resume=True)
        == tmp_path / "checkpoint-10"
    )


def test_explicit_resume_checkpoint_must_exist(tmp_path: Path) -> None:
    missing = tmp_path / "checkpoint-404"

    try:
        resolve_resume_checkpoint(tmp_path, resume_from_checkpoint=str(missing))
    except FileNotFoundError as exc:
        assert "resume checkpoint does not exist" in str(exc)
    else:
        raise AssertionError("expected missing checkpoint to fail")


def test_sft_dry_run_writes_recoverable_artifacts(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path)

    result = run_sft(config_path=config_path, dry_run=True)

    run_dir = Path(result["run_dir"])
    metadata_path = run_dir / "run_metadata.json"
    assert result["status"] == "dry_run_complete"
    assert metadata_path.exists()
    assert (run_dir / "checkpoint-dry-run" / "metadata.json").exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["dry_run"] is True
    assert metadata["verifier_version"] == VERIFIER_VERSION
    assert Path(metadata["manifest_path"]).exists()
    assert Path(metadata["train_jsonl_path"]).exists()


def test_raw_output_evaluation_report_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_split_manifest(), manifest_path)
    raw_outputs_path = tmp_path / "raw.jsonl"
    write_jsonl(
        [
            {
                "id": "ok",
                "numbers": [8, 2, 7, 3],
                "target": 24,
                "output": "<answer>((8 - 2) * (7 - 3))</answer>",
            },
            {
                "id": "bad-value",
                "numbers": [8, 2, 7, 3],
                "target": 24,
                "output": "<answer>8 + 2 + 7 + 3</answer>",
            },
            {
                "id": "bad-format",
                "numbers": [8, 2, 7, 3],
                "target": 24,
                "output": "((8 - 2) * (7 - 3))",
            },
        ],
        raw_outputs_path,
    )

    report = evaluate_raw_outputs_file(
        raw_outputs_path=raw_outputs_path,
        report_path=tmp_path / "report.json",
        model_name="unit-test-model",
        checkpoint="checkpoint-1",
        split_manifest=manifest_path,
        split="validation",
        decoding=DecodingConfig(),
    )

    assert report["schema_version"] == "game24-eval-report-v1"
    assert report["verifier_version"] == VERIFIER_VERSION
    assert report["answer_contract"] == "<answer>...</answer>"
    assert report["metrics"]["total"] == 3
    assert report["metrics"]["format_rate"] == 2 / 3
    assert report["metrics"]["valid_expr_rate"] == 2 / 3
    assert report["metrics"]["solve_rate"] == 1 / 3


def test_solver_dry_run_evaluation_is_perfect_on_solvable_split(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_split_manifest(), manifest_path)

    report = evaluate_solver_dry_run(
        manifest_path=manifest_path,
        output_dir=tmp_path / "eval",
        limit=8,
    )

    assert report["metrics"]["total"] == 8
    assert report["metrics"]["solve_rate"] == 1.0
    assert (tmp_path / "eval" / "validation-raw-outputs.jsonl").exists()
    assert (tmp_path / "eval" / "validation-eval-report.json").exists()


def _write_test_config(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "splits" / "manifest.json"
    train_jsonl_path = tmp_path / "sft" / "train.jsonl"
    output_dir = tmp_path / "outputs"
    config = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "method": "lora_sft",
        "task": "standard_game24",
        "data": {
            "split_seed": 20260613,
            "traces_per_puzzle": 1,
            "manifest_path": str(manifest_path),
            "train_jsonl_path": str(train_jsonl_path),
        },
        "training": {
            "output_dir": str(output_dir),
            "run_name": "unit-test",
            "max_length": 128,
            "eos_token": "<|im_end|>",
            "save_steps": 7,
        },
        "evaluation": {
            "do_sample": False,
            "max_new_tokens": 64,
            "success_gate": 0.7,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
